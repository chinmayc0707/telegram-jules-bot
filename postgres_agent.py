from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openrouter import ChatOpenRouter
from langgraph.prebuilt import create_react_agent
from psycopg import connect
import os
from dotenv import load_dotenv

load_dotenv()

class PersistentAgent:
    def __init__(llm,DB_URI = os.getenv('DB_URI'),llm=ChatOpenRouter(model="openrouter/free",api_key=os.getenv('OPEN_ROUTER_API'))):
       self.conn = connect(DB_URI)
       self.conn.autocommit = True
       self.checkpointer = PostgresSaver(conn)
       self.checkpointer.setup()
       self.agent=create_react_agent(llm,tools=[],checkpointer=checkpointer)
    def chat(prompt,thread_id):
        return agent.invoke({"messages": [
                {"role":"user","content":prompt}
            ]},
            config={"thread_id":"user1"})['messages'][-1].content
