from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from psycopg import connect
import os
import logging
import tiktoken
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class PersistentAgent:
    def __init__(
        self,
        db_uri=os.getenv("DB_URI"),
        llm=None,
        tools=[],
        context_limit: int = 8192,
        preserve_last_k: int = 4,
        system_message: str = "You are a helpful AI assistant.",
    ):
        """
        Args:
            db_uri:          PostgreSQL connection string.
            llm:             LangChain-compatible LLM instance.
            context_limit:   Maximum token context window of the model.
            preserve_last_k: Number of recent messages to keep untouched
                             during summarization.
            system_message:  Initial system instructions for the agent.
        """
        if llm is None:
            llm = ChatOpenRouter(
                model="openrouter/free",
                api_key=os.getenv("OPEN_ROUTER_API"),
            )

        self.llm = llm
        self.context_limit = context_limit
        self.preserve_last_k = preserve_last_k
        self.summarization_threshold = int(context_limit * 0.75)
        self.tools=tools
        self.system_message = system_message
        # ── Postgres checkpoint setup ────────────────────────────────
        self.conn = connect(db_uri)
        self.conn.autocommit = True
        self.checkpointer = PostgresSaver(self.conn)
        self.checkpointer.setup()

        # ── Agent ────────────────────────────────────────────────────
        self.agent = create_react_agent(
            llm, tools=self.tools, checkpointer=self.checkpointer, prompt=self.system_message
        )

        # ── Token encoder (cl100k_base works for most modern models) ─
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoder = None

    # ── token helpers ────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """Return the approximate token count for *text*."""
        if self._encoder:
            return len(self._encoder.encode(text))
        # Rough fallback: ~4 chars per token
        return len(text) // 4

    def _messages_token_count(self, messages) -> int:
        """Sum of token counts across all messages."""
        total = 0
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else str(msg)
            total += self._count_tokens(content)
        return total

    # ── summarization ────────────────────────────────────────────────

    def _summarize_if_needed(self, messages: list) -> list:
        """
        Check whether the conversation has crossed 75% of context_limit.
        If so, summarize all messages *except* the last ``preserve_last_k``
        into a single SystemMessage summary and return the trimmed list.
        Otherwise return the original messages unchanged.
        """
        token_count = self._messages_token_count(messages)

        if token_count <= self.summarization_threshold:
            return messages          # within budget – nothing to do

        # Determine the split point
        k = min(self.preserve_last_k, len(messages))
        if k >= len(messages):
            # Everything is "recent"; nothing to summarize
            return messages

        older_messages = messages[:-k]
        recent_messages = messages[-k:]

        # Build a readable transcript of the older messages
        transcript_lines = []
        for msg in older_messages:
            role = getattr(msg, "type", "unknown")
            content = msg.content if hasattr(msg, "content") else str(msg)
            transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)

        # Ask the LLM to summarize the transcript
        summary_prompt = (
            "You are a concise summarizer. Below is a conversation history. "
            "Produce a short, factual summary capturing all important details, "
            "user preferences, decisions, and any information that would be "
            "needed to continue the conversation naturally.\n\n"
            f"--- CONVERSATION ---\n{transcript}\n--- END ---\n\n"
            "Summary:"
        )

        summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
        summary_text = (
            summary_response.content
            if hasattr(summary_response, "content")
            else str(summary_response)
        )

        # Compose the condensed message list
        summary_msg = SystemMessage(
            content=(
                "[Conversation Summary]\n"
                f"{summary_text}"
            )
        )
        return [summary_msg] + list(recent_messages)

    # ── orphaned tool-call repair ────────────────────────────────────

    def _repair_orphaned_tool_calls(self, messages: list) -> list:
        """
        Scan the message history for AIMessages whose ``tool_calls`` have
        no matching ToolMessage in the messages that follow.  For every
        orphan, inject a synthetic ToolMessage with an error payload so
        that LangGraph/LLM providers don't reject the history.

        Returns a new list (the original is not mutated).
        """
        # Collect IDs of all existing ToolMessages
        existing_tool_msg_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                existing_tool_msg_ids.add(msg.tool_call_id)

        repaired: list = []
        patched = False

        for msg in messages:
            repaired.append(msg)

            # Only AIMessages can carry tool_calls
            if not isinstance(msg, AIMessage):
                continue

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                continue

            for tc in tool_calls:
                tc_id = tc.get("id") or tc.get("tool_call_id", "")
                if tc_id and tc_id not in existing_tool_msg_ids:
                    # Orphan detected – inject a placeholder ToolMessage
                    logger.warning(
                        "Repairing orphaned tool call %s (%s)",
                        tc_id,
                        tc.get("name", "unknown"),
                    )
                    repaired.append(
                        ToolMessage(
                            content=(
                                "[Error] Tool call failed before producing a "
                                "result. The error has been handled; you may "
                                "retry or continue the conversation."
                            ),
                            tool_call_id=tc_id,
                        )
                    )
                    existing_tool_msg_ids.add(tc_id)
                    patched = True

        if patched:
            logger.info(
                "Repaired %d orphaned tool call(s) in message history.",
                sum(1 for m in repaired if isinstance(m, ToolMessage)
                    and "[Error]" in m.content),
            )
        return repaired

    # ── public API ───────────────────────────────────────────────────

    def chat(self, prompt: str, thread_id: str) -> str:
        """
        Send *prompt* to the agent under *thread_id*.

        Before invoking, the stored message history is checked for
        orphaned tool calls (repaired automatically) and summarized
        when it exceeds 75 % of the model context window.
        """
        config = {"configurable": {"thread_id": thread_id}}

        # ── 1. Retrieve existing history from the checkpoint ─────────
        state = self.agent.get_state(config)
        existing_messages = (
            state.values.get("messages", []) if state and state.values else []
        )

        if existing_messages:
            dirty = False

            # ── 2a. Repair orphaned tool calls ───────────────────────
            repaired = self._repair_orphaned_tool_calls(existing_messages)
            if len(repaired) != len(existing_messages):
                existing_messages = repaired
                dirty = True

            # ── 2b. Summarize if over budget ─────────────────────────
            trimmed = self._summarize_if_needed(existing_messages)
            if len(trimmed) != len(existing_messages):
                existing_messages = trimmed
                dirty = True

            # ── 2c. Persist repairs / summarization ──────────────────
            if dirty:
                self.agent.update_state(
                    config, {"messages": existing_messages}
                )

        # ── 3. Invoke the agent with the new user message ────────────
        try:
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )
            return result["messages"][-1].content

        except Exception as e:
            logger.error("Agent invocation failed: %s", e)

            # The checkpoint may now contain an AIMessage with tool_calls
            # but no ToolMessage.  Repair it so the *next* call doesn't
            # hit INVALID_CHAT_HISTORY.
            try:
                state = self.agent.get_state(config)
                msgs = (
                    state.values.get("messages", [])
                    if state and state.values
                    else []
                )
                repaired = self._repair_orphaned_tool_calls(msgs)
                if len(repaired) != len(msgs):
                    self.agent.update_state(
                        config, {"messages": repaired}
                    )
                    logger.info(
                        "Checkpoint repaired after failed invocation."
                    )
            except Exception as repair_err:
                logger.error(
                    "Failed to repair checkpoint: %s", repair_err
                )

            return f"Sorry, an error occurred: {e}"


if __name__=="__main__":
    import jules_api
    def evaluate(expression:str):
        """Python's inbuilt eval function"""
        return eval(expression)
    agent=PersistentAgent(system_message="You are a telegram messenger. Telegram doesn't have rendering for markdown. Do not include any formatting in your response like bold, bullets etc",llm=ChatOllama(model="gemma4:e2b"),tools=[eval],context_limit=1_28_000)
    print(agent.chat(input("prompt: "),"user1"))