import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from sqlalchemy import text
import pandas as pd
import re
from dotenv import load_dotenv

load_dotenv()


def create_agent(engine):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-20b",
        temperature=0,
    )
    db = SQLDatabase(engine)
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        max_iterations=15,
        handle_parsing_errors=True,
        max_execution_time=60,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )
    return agent_executor, db


def run_query(agent_executor, question: str) -> dict:
    try:
        enriched_question = f"""
        You are a SQL expert. The database has a table called 'data'.
        First use tools to check the schema, then answer this question: {question}
        Give a clear final answer based on the query results.
        """
        result = agent_executor.invoke({"input": enriched_question})
        return {
            "answer": result.get("output", "No answer found."),
            "success": True
        }
    except Exception as e:
        error_msg = str(e)
        if "Could not parse LLM output" in error_msg:
            match = re.search(
                r"Could not parse LLM output: `(.+?)`",
                error_msg,
                re.DOTALL
            )
            if match:
                return {
                    "answer": match.group(1).strip(),
                    "success": True
                }
        return {
            "answer": f"Error: {error_msg}",
            "success": False
        }


def extract_sql_and_run(engine, question: str, llm) -> tuple:
    db = SQLDatabase(engine)
    schema = db.get_table_info()
    prompt = f"""
You are a SQLite expert. Given the following schema:
{schema}

Write ONLY a valid SQLite SQL query to answer this question:
"{question}"

STRICT RULES:
- Output ONLY the raw SQL query
- No markdown, no backticks, no explanation
- Just the SQL starting with SELECT
"""
    response = llm.invoke(prompt)
    sql_query = response.content.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    lines = sql_query.split('\n')
    sql_lines = [
        line for line in lines
        if not line.strip().startswith('--')
        and not line.strip().lower().startswith('here')
        and not line.strip().lower().startswith('this')
    ]
    sql_query = '\n'.join(sql_lines).strip()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql_query), conn)
    return sql_query, df