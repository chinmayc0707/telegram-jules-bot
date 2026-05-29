import psycopg
import uuid
from langchain_postgres import PostgresChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openrouter import ChatOpenRouter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class ChatMemoryManager:
    def __init__(self, database_url: str, model_name: str = "gemma-4-31b-it:free"):
        self.database_url = database_url
        self.table_name = "chat_memory"
        
        # 1. Initialize LLM
        self.llm = ChatOpenRouter(
            model=model_name,
            temperature=0
        )
        
        # 2. Setup Prompt and Chain
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        self.chain = self.prompt | self.llm
        
        # 3. Create Memory Wrapper
        self.memory_chain = RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        
        # 4. Ensure database tables exist
        self._create_tables()

    def _create_tables(self):
        """Creates the necessary tables in PostgreSQL if they don't exist."""
        with psycopg.connect(self.database_url) as conn:
            PostgresChatMessageHistory.create_tables(conn, self.table_name)

    def _get_session_history(self, session_id: str):
        """Internal helper to provide session history to RunnableWithMessageHistory."""
        # Convert a normal string to a deterministic UUID (UUID v5)
        # This allows us to use normal strings as session IDs while satisfying
        # the database's UUID requirement.
        NAMESPACE_DNS = uuid.NAMESPACE_DNS
        deterministic_uuid = str(uuid.uuid5(NAMESPACE_DNS, session_id))
        
        # Note: langchain-postgres requires a connection object, not a string
        conn = psycopg.connect(self.database_url)
        return PostgresChatMessageHistory(
            self.table_name,
            deterministic_uuid,
            sync_connection=conn
        )

    def chat(self, user_input: str,session_id: str):
        """
        Sends a message to the AI and maintains persistent memory.
        Args:
            session_id (str): A string.
            user_input (str): The user's message.
        """
        response = self.memory_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content

    def clear_session(self, session_id: str):
        """Clears the chat history for a specific session."""
        history = self._get_session_history(session_id)
        history.clear()

    def delete_session(self, session_id: str):
        """
        Permanently deletes all records for a specific session from the database.
        """
        # Convert the normal string to the same deterministic UUID used for storage
        NAMESPACE_DNS = uuid.NAMESPACE_DNS
        deterministic_uuid = str(uuid.uuid5(NAMESPACE_DNS, session_id))
        
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE session_id = %s", 
                    (deterministic_uuid,)
                )
            conn.commit()

# --- Example Usage ---
if __name__ == "__main__":
    # supabase connection pooler link
    DATABASE_URL = "postgresql://postgres.rskdxiqyevrkgozeuqhx:Chinmayc0707@aws-1-ap-northeast-1.pooler.supabase.com:6543/postgres"
    
    # Initialize the manager
    manager = ChatMemoryManager(DATABASE_URL)
    
    # Now you can use any normal string!
    session_id = "convo1"
    
    # First interaction
    print("User: Hi, my name is Chinmay.")
    res1 = manager.chat(session_id, "how do u remember that")
    print(f"AI: {res1}\n")
    
    # # Second interaction (testing memory)
    # print("User: What is my name?")
    # res2 = manager.chat(session_id, "What is my name?")
    # print(f"AI: {res2}")
