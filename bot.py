import os
import re
import logging
from contextlib import asynccontextmanager
from postgres_agent import PersistentAgent
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_openrouter import ChatOpenRouter
from jules_api import list_jules_sources,create_jules_session,get_jules_session_status,approve_jules_plan,send_jules_message
load_dotenv()


def strip_markdown(text: str) -> str:
    """Remove common markdown formatting so Telegram shows plain text."""
    # Code blocks (``` ... ```)
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group().strip("`").strip(), text)
    # Inline code (`...`)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold + italic (***text*** or ___text___)
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)
    # Bold (**text** or __text__)
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # Italic (*text* or _text_)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    # Strikethrough (~~text~~)
    text = re.sub(r"~~(.+?)~~", r"\1", text)
    # Headers (# ... ##  etc.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Links [text](url) → text (url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    # Images ![alt](url) → alt (url)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1 (\2)", text)
    # Bullet points (- or * at line start)
    text = re.sub(r"^[\-\*]\s+", "• ", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\*{3,}$", "", text, flags=re.MULTILINE)
    return text.strip()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Telegram helpers ─────────────────────────────────────────────────────────


async def set_webhook() -> dict:
    """Register the webhook URL with Telegram."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TELEGRAM_API}/setWebhook",
            json={"url": f"{WEBHOOK_URL}/webhook"},
        )
        data = response.json()
        logger.info("setWebhook response: %s", data)
        return data


async def send_message(chat_id: int, text: str) -> dict:
    """Send a text message to a Telegram chat."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )
        data = response.json()
        logger.info("sendMessage response: %s", data)
        return data


# ── FastAPI app ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup, register the webhook with Telegram."""
    await set_webhook()
    yield


app = FastAPI(title="Telegram Echo Bot", lifespan=lifespan)


@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming Telegram updates and echo the message back."""
    update = await request.json()
    logger.info("Received update: %s", update)
    tools=[list_jules_sources, 
        create_jules_session, 
        get_jules_session_status, 
        approve_jules_plan, 
        send_jules_message]
    agent=PersistentAgent(tools=tools,system_message="You are an expert developer agent capable of using the Jules API to automate software development tasks in GitHub repositories. You can list sources, create sessions, check status, and interact with the agent.",context_limit=1_28_000)
    message = update.get("message")
    if message:
        chat_id = message["chat"]["id"]
        text = message.get("text", "")

        if text == "/clear":
            agent.clear_chat(str(chat_id))
            await send_message(chat_id, "Chat history cleared.")
        elif text:
            response=agent.chat(text,str(chat_id))
            # Strip markdown so Telegram displays clean plain text
            await send_message(chat_id, strip_markdown(response))

    return JSONResponse(content={"ok": True})


@app.get("/health")
async def health():
    """Simple health-check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bot:app", host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), reload=True)
