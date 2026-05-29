import os
import logging
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

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

    message = update.get("message")
    if message:
        chat_id = message["chat"]["id"]
        text = message.get("text", "")

        if text:
            # Echo the received text back to the sender
            await send_message(chat_id, text)

    return JSONResponse(content={"ok": True})


@app.get("/health")
async def health():
    """Simple health-check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bot:app", host="0.0.0.0", port=8000, reload=True)
