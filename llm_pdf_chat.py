from langchain import hub
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain.prompts import ChatPromptTemplate
from typing import Literal
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()
travily_web_search = TavilySearchResults()


def get_vectorstore(files):
    """Converts the uploaded pdf documents into embeddings and stores them in a vectorstore"""
    loader = PyPDFLoader(files)
    docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap = 100))
    vectorstore = FAISS.from_documents(docs,OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever
"""
class RouteQuery(BaseModel):
    Route a user query to the most relevant datasource.

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )
"""
class web_search_tool(BaseModel):
    """
    The internet. Use web_search for questions that are related to anything else than agents, prompt engineering, and adversarial attacks.
    """

    query: str = Field(description="The query to use when searching the internet.")


class vectorstore(BaseModel):
    """
    A vectorstore containing documents. Use the vectorstore for questions on these topics.
    """

    query: str = Field(description="The query to use when searching the vectorstore.")

def history_aware_prompt(question,chat_history):
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)    
    history_aware_chain = contextualize_q_prompt | llm | StrOutputParser()
    return history_aware_chain.invoke({"question":question,"chat_history":chat_history})

def get_rag_chain():
    """Returns a RAG chain"""
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain


def get_router_chain():
    """Returns a Router chain with the tools web_search and vectorstore"""
    # LLM with function call
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_llm_router = llm.bind_tools(tools=[web_search_tool, vectorstore])

    # Prompt
    system ="""You are an expert at routing a user question to either a vectorstore or a web search.
    The vectorstore contains a diverse set of documents. Use the vectorstore if it is likely to contain the information needed to answer the question.
    Otherwise, use web search."""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router
    return question_router

class GradeDocuments(BaseModel):
    """Binary Score for relevance check on retreived documents"""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no' "
    )


def get_relevance_grader_chain():
    """Returns a chain which checks that whether a document is relevant or onot"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system ="""You are a grader assessing relevance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )
    grader = grade_prompt | structured_llm_grader
    return grader


class GradeHallucinations(BaseModel):
    """Binary score for hallucinations present in the generated answer"""
    binary_score: str = Field(
        description = "Answer grounded in facts, 'yes' or 'no' "
    )


def get_hallucination_grader_chain():
    """Returns a chain which checks for hallucinations"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether the LLM generation is grounded in the provided documents. 
If the generation aligns with the facts presented in the documents and does not introduce unsupported information, grade it as 'yes'. 
Otherwise, grade it as 'no'."""
    
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    grader = grade_prompt | structured_llm_grader
    return grader

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

def get_answer_grader_chain():
    """Returns a chain which checks whether a question is answered correctly by the LLM"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    # Prompt
    system ="""You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    grader = grade_prompt | structured_llm_grader
    return grader


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]



def retrieve(state,retriever):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state
        retriever(VectorStoreRetriever): The retriever to use for fetching documents

    Returns:
        dict: New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def llm_fallback(state):
    """
    Generate answer using the LLM w/o vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---LLM Fallback---")
    question = state["question"]
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate()
    llm_chain = prompt | llm | StrOutputParser()
    generation = llm_chain.invoke({"question": question})
    return {"question": question, "generation": generation}


def generate(state):
    """
    Generate answer using the vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")

    rag_chain = get_rag_chain()

    question = state["question"]
    documents = state["documents"]
    if not isinstance(documents, list):
        documents = [documents]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    retrieval_grader = get_relevance_grader_chain()
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def get_query_rewrite_chain():
    """Rewrites a Question into a prompt suitable for search engine queries"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    system = """You are an expert query optimizer. 
    Your task is to rewrite an input question to be perfectly tailored for search engines, focusing on keyword optimization, clarity, and relevance. 
    If the original question is already optimal, leave it unchanged. Your goal is to enhance search precision and relevance without altering the core intent of the query."""
    
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question optimized for search engines.",
            ),
        ]
    )
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    rewrite_prompt = get_query_rewrite_chain()
    # Web search
    new_question = rewrite_prompt.invoke({"question":question})

    docs = travily_web_search.invoke({"query": new_question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}


### Edges ###


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    question_router = get_router_chain()
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})

    # Fallback to LLM or raise error if no decision
    if "tool_calls" not in source.additional_kwargs:
        print("---ROUTE QUESTION TO LLM---")
        return "llm_fallback"
    if len(source.additional_kwargs["tool_calls"]) == 0:
        raise "Router could not decide source"

    # Choose datasource
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    else:
        print("---ROUTE QUESTION TO LLM---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, WEB SEARCH---")
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state, max_retries=10):
    """
    Determines whether the generation is grounded in the document and answers the question.
    Args:
        state (dict): The current graph state
        max_retries (int): Maximum number of retries before termination
    Returns:
        str: Decision for the next node to call
    """
    # Retrieve the current retry count from the state
    retry_count = state.get('retry_count', 0)
    retry_count += 1
    print(f"---CHECK HALLUCINATIONS--- (Retry {retry_count})")

    hallucination_grader = get_hallucination_grader_chain()
    answer_grader = get_answer_grader_chain()
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Check for hallucinations
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check if the generation addresses the question
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print(f"---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RETRY {retry_count}/{max_retries}---")
        if retry_count >= max_retries:
            print("---MAX RETRIES REACHED, FALLING BACK---")
            return "not useful"
        
        # Update the state with the incremented retry count
        state["retry_count"] = retry_count
        return "retry"

def graph(retriever):
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("web_search", web_search)  # web search
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # rag
    workflow.add_node("llm_fallback", llm_fallback)  # llm

    # Build graph
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
            "llm_fallback": "llm_fallback",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",  # Hallucinations: re-generate
            "not useful": "web_search",  # Fails to answer the question: fallback to web search
            "useful": END,
            "retry": "generate",  # Retry logic should point back to the generate node
        },
    )
    workflow.add_edge("llm_fallback", END)

    # Compile
    app = workflow.compile()
    return app
