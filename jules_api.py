import requests
import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class JulesClient:
    """Client for the Google Jules API."""

    BASE_URL = "https://jules.googleapis.com/v1alpha"
    _instance: Optional["JulesClient"] = None

    def __new__(cls, *args, **kwargs):
        """Singleton – reuse the same client across tool calls."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, api_key: Optional[str] = None):
        if hasattr(self, "_initialized"):
            return
        self.api_key = api_key or os.getenv("JULES_API_KEY")
        if not self.api_key:
            raise ValueError(
                "JULES_API_KEY environment variable is not set and no api_key provided."
            )
        self.headers = {
            "X-Goog-Api-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._initialized = True

    def _request(self, method, endpoint, params=None, data=None):
        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        resp = requests.request(
            method, url, headers=self.headers, params=params, json=data
        )
        resp.raise_for_status()
        return resp.json()

    # ── raw API methods ──────────────────────────────────────────────

    def list_sources(self):
        return self._request("GET", "sources")

    def create_session(
        self,
        prompt,
        source_name,
        branch,
        title=None,
        automation_mode="AUTO_CREATE_PR",
        require_plan_approval=False,
    ):
        data = {
            "prompt": prompt,
            "sourceContext": {
                "source": source_name,
                "githubRepoContext": {"startingBranch": branch},
            },
            "automationMode": automation_mode,
            "title": title or prompt,
            "requirePlanApproval": require_plan_approval,
        }
        return self._request("POST", "sessions", data=data)

    def list_sessions(self, page_size=10):
        return self._request("GET", "sessions", params={"pageSize": page_size})

    def get_session(self, session_id):
        return self._request("GET", f"sessions/{session_id}")

    def approve_plan(self, session_id):
        return self._request("POST", f"sessions/{session_id}:approvePlan")

    def send_message(self, session_id, message):
        data = {"message": message}
        return self._request(
            "POST", f"sessions/{session_id}:sendMessage", data=data
        )


# ── helpers for clean LLM output ─────────────────────────────────────────


def _clean_id(name: str) -> str:
    """'sessions/abc123' → 'abc123'"""
    return name.rsplit("/", 1)[-1] if "/" in name else name


def _fmt_source(src: dict) -> str:
    name = src.get("name", "?")
    repo = src.get("githubRepoContext", {}).get("repoFullName", "")
    return f"• {name}" + (f"  ({repo})" if repo else "")


def _fmt_session_summary(s: dict) -> str:
    """One-line session digest the LLM can reason over."""
    sid = _clean_id(s.get("name", "?"))
    status = s.get("status", "UNKNOWN")
    title = s.get("title", "")
    pr = s.get("pullRequestUrl", "")
    parts = [f"id={sid}", f"status={status}"]
    if title:
        parts.append(f"title={title}")
    if pr:
        parts.append(f"pr={pr}")
    return " | ".join(parts)


# ── tool functions (exposed to the LLM) ──────────────────────────────────


def list_jules_sources() -> str:
    """List all GitHub repositories connected to Jules."""
    data = JulesClient().list_sources()
    sources = data.get("sources", [])
    if not sources:
        return "No sources connected."
    lines = [_fmt_source(s) for s in sources]
    return "Connected sources:\n" + "\n".join(lines)


def create_jules_session(
    prompt: str,
    source_name: str,
    branch: str = "main",
    title: str = "",
) -> str:
    """Create a Jules session to run a task on a repo.

    Args:
        prompt: Detailed instruction for Jules (what to build/fix).
        source_name: Source identifier, e.g. 'sources/github/owner/repo'.
        branch: Starting branch (default 'main').
        title: Short title for the session.
    """
    s = JulesClient().create_session(
        prompt=prompt,
        source_name=source_name,
        branch=branch,
        title=title or None,
    )
    return f"Session created: {_fmt_session_summary(s)}"


def get_jules_session_status(session_id: str) -> str:
    """Check the current status of a Jules session.

    Args:
        session_id: The session ID (with or without 'sessions/' prefix).
    """
    sid = session_id.replace("sessions/", "")
    s = JulesClient().get_session(sid)

    status = s.get("status", "UNKNOWN")
    title = s.get("title", "")
    pr = s.get("pullRequestUrl", "")
    plan = s.get("plan", {}).get("description", "")

    lines = [f"Session {sid}: {status}"]
    if title:
        lines.append(f"Title: {title}")
    if plan:
        lines.append(f"Plan: {plan}")
    if pr:
        lines.append(f"PR: {pr}")
    return "\n".join(lines)


def approve_jules_plan(session_id: str) -> str:
    """Approve the plan so Jules starts writing code.

    Args:
        session_id: The session ID (with or without 'sessions/' prefix).
    """
    sid = session_id.replace("sessions/", "")
    s = JulesClient().approve_plan(sid)
    return f"Plan approved for session {sid}. Status: {s.get('status', 'UNKNOWN')}"


def send_jules_message(session_id: str, message: str) -> str:
    """Send follow-up instructions or feedback to an active Jules session.

    Args:
        session_id: The session ID (with or without 'sessions/' prefix).
        message: The feedback or instruction text.
    """
    sid = session_id.replace("sessions/", "")
    JulesClient().send_message(sid, message)
    return f"Message sent to session {sid}."


if __name__ == "__main__":
    import postgres_agent
    from langchain_openrouter import ChatOpenRouter

    jules_tools = [
        list_jules_sources,
        create_jules_session,
        get_jules_session_status,
        approve_jules_plan,
        send_jules_message,
    ]

    agent = postgres_agent.PersistentAgent(
        tools=jules_tools,
        llm=ChatOpenRouter(
            model="google/gemma-4-31b-it:free",
            api_key=os.getenv("OPEN_ROUTER_API"),
        ),
        system_message=(
            "You are a developer agent that uses the Jules API to automate "
            "software tasks on GitHub repos. You can list sources, create "
            "sessions, check status, approve plans, and send messages."
        ),
    )

    print(agent.chat(input("prompt: "), "user2"))
