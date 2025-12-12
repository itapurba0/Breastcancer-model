"""Chat interface helpers.

This module exposes a small wrapper that `api.py` can call. It keeps the API
file focused and lets you implement a full chatbot separately under
`backend/chatbot/`.
"""
import os
from datetime import datetime
from typing import Optional


BASE_DIR = os.path.dirname(__file__)


def call_chatbot(message: str, session_id: Optional[str] = None) -> str:
    try:
        # dynamic import of an optional chatbot implementation
        from . import chatbot as chatbot_pkg  # type: ignore
        if hasattr(chatbot_pkg, "reply"):
            return chatbot_pkg.reply(message, session_id=session_id)
        if hasattr(chatbot_pkg, "generate_reply"):
            return chatbot_pkg.generate_reply(message, session_id=session_id)
        if hasattr(chatbot_pkg, "chat"):
            return chatbot_pkg.chat(message, session_id=session_id)
    except Exception:
        pass
    # fallback reply
    return f"I received: '{message[:240]}' -- (placeholder reply)."


def log_chat(message: str, reply: str, session_id: Optional[str]):
    try:
        log_path = os.path.join(BASE_DIR, "chat_logs.txt")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.utcnow().isoformat()}\t{session_id or '-'}\t{message}\t{reply}\n")
    except Exception:
        pass
