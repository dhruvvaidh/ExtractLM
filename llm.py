# Tracing
import os
import sqlite3
from dotenv import load_dotenv
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.agent_toolkits import create_sql_agent,SQLDatabaseToolkit

load_dotenv()
filepath = "cars.csv"

def csv_to_db(conn,dbname:str,filepath):
    table_name = os.path.splitext(os.path.basename(filepath))[0]
    df = pd.read_csv(filepath)
    with sqlite3.connect(dbname) as conn:
        df.to_sql(name = table_name, 
                con = conn, 
                index=False, 
                if_exists="replace")
    #conn.close()
    return SQLDatabase.from_uri("sqlite:///"+dbname)



# Task Decomposition
data_dictionary = """
- Dimensions.Height: Height of the car in centimeters
- Dimensions.Length: Length of the car in centimeters
- Dimensions.Width: Width of the car in centimeters
- Engine Information.Driveline: Type of driveline (e.g., All-wheel drive, Front-wheel drive)
- Engine Information.Engine Type: Details about the engine type (e.g., Audi 3.2L 6 cylinder 250hp 236ft-lbs)
- Engine Information.Hybrid: Indicates if the car is a hybrid (True/False)
- Engine Information.Number of Forward Gears: Number of forward gears in the car
- Engine Information.Transmission: Type of transmission (e.g., 6 Speed Automatic Select Shift)
- Fuel Information.City mpg: City mileage in miles per gallon
- Fuel Information.Fuel Type: Type of fuel used (e.g., Gasoline)
- Fuel Information.Highway mpg: Highway mileage in miles per gallon
- Identification.Classification: Type of transmission (e.g., Automatic transmission)
- Identification.ID: Unique identifier for the car (e.g., 2009 Audi A3 3.2)
- Identification.Make: Manufacturer of the car (e.g., Audi)
- Identification.Model Year: Model year of the car (e.g., 2009)
- Identification.Year: Year of manufacture
- Engine Information.Engine Statistics.Horsepower: Horsepower of the engine
- Engine Information.Engine Statistics.Torque: Torque of the engine
"""

def decompose_prompt(question:str,data_dictionary = data_dictionary):
    template = """
    You are an assistant specialized in writing SQL statements to extract complex insights from data.
    The dataset contains the following columns:
    {data_dictionary}

    Given an input question, your task is to break it down into multiple sub-questions that can be answered in isolation, 
    leveraging the context of the data provided. The goal is to generate multiple sub-questions 
    that collectively help in answering the input question comprehensively.
    Generate Multiple Search Queries for related to: {question}

    Output(2 queries):
    """

    prompt_decomposition = ChatPromptTemplate.from_template(template)

    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)

    # Chain
    generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))

    # Run
    #question = "Top 10 cars with 2L engine and german make"
    questions = generate_queries_decomposition.invoke({"data_dictionary":data_dictionary,"question":question})

    return questions

def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

def qa_pairs(questions,db):
    llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    # Creating an SQL agent which interacts with the database
    agent_executor = create_sql_agent(llm, toolkit=toolkit, agent_type="openai-tools", verbose=False)
    q_a_pairs = ""
    for q in questions:

        answer = agent_executor.invoke(q)
        q_a_pair = format_qa_pair(q,answer['output'])
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

    return q_a_pairs

def compile_answers(q_a_pairs,question):
    template = """

    Here is a set of question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Use these to synthesize an answer to the question: \n {question}
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    decomposition_prompt = ChatPromptTemplate.from_template(template)
    final_chain = decomposition_prompt | llm | StrOutputParser()

    final_output = final_chain.invoke({'q_a_pairs':q_a_pairs,'question':question})
    return final_output



