import getpass
import os
import argparse
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool


PARSER = argparse.ArgumentParser(description="command line flags.")
PARSER.add_argument('--user', type=str, help='db user')
PARSER.add_argument('--pwd', type=str, help='db password')
PARSER.add_argument('--db', type=str, help='db name')
ARGS = PARSER.parse_args()
LLM_VERSION = "gemini-2.0-flash"
LLM_PROVIDER = "google_genai"


class ModelAdapter:
    def _init_llm(self):
        if not os.environ.get("GOOGLE_API_KEY"):
            raise Exception("API key missing for Google Gemini")
        self.model = init_chat_model(LLM_VERSION, model_provider=LLM_PROVIDER)
        self.db = SQLDatabase.from_uri(f"postgresql://{ARGS.user}:{ARGS.pwd}@localhost:5432/{ARGS.db}")

    def _init_prompt_config(self):
        self.system_template = """
            Given an input question, create a syntactically correct {dialect} query to
            run to help find the answer. Unless the user specifies in his question a
            specific number of examples they wish to obtain, always limit your query to
            at most {top_k} results. You can order the results by a relevant column to
            return the most interesting examples in the database.

            Never query for all the columns from a specific table, only ask for a the
            few relevant columns given the question.

            Pay attention to use only the column names that you can see in the schema
            description. Be careful to not query for columns that do not exist. Also,
            pay attention to which column is in which table.

            Only use the following tables:
            {table_info}
            """
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_template), ("user", "Question: {input}")]
        )
    def __init__(self):
        self._init_llm()
        self._init_prompt_config()

    class State(TypedDict):
        question: str
        query: str
        result: str
        answer: str

    class DBQuery(TypedDict):
        """Generated SQL query."""
        query: Annotated[str, ..., "Valid SQL query."]

    def build_query(self, state:State, k):
        prompt = self.prompt_template.invoke({"dialect": self.db.dialect, "top_k": k, "table_info": self.db.get_table_info(), "input": state["question"]})
        self.structured_model = self.model.with_structured_output(self.DBQuery)
        response = self.structured_model.invoke(prompt)
        print(f"querying {response['query']}")
        return {**state, "query": response["query"]}

    def execute_query(self, state: State):
        self.execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        return {**state, "result": self.execute_query_tool.invoke(state["query"])}

    def invoke(self, state:State):
        prompt = (
            "Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n\n"
            f"Question: {state['question']}\n"
            f"SQL Query: {state['query']}\n"
            f"SQL Result: {state['result']}"
        )
        response = self.model.invoke(prompt)
        return {**state, "answer": response.content}


adapter = ModelAdapter()
response = adapter.invoke(adapter.execute_query(adapter.build_query({"question":"How many products are there?"},10)))
print(response['answer'])