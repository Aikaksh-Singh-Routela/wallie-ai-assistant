"""YouTube live chat monitor.

Uses the YouTube Data API v3 liveChatMessages.list endpoint with OAuth. The user
drops a client_secret.json under scripts/ and runs the first auth flow once; the
refresh token is cached in token.json.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from loguru import logger

from .base import ChatMessage, ChatMonitor


class YouTubeChatMonitor(ChatMonitor):
    platform = "youtube"

    def __init__(self, *, client_secret_file: str, live_chat_id: str, api_key: str = "") -> None:
        self._client_secret_file = client_secret_file
        self._live_chat_id = live_chat_id
        self._api_key = api_key
        self._service = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self, out_queue: asyncio.Queue[ChatMessage]) -> None:
        if not self._live_chat_id:
            logger.warning("youtube: no live_chat_id configured; monitor disabled")
            return
        try:
            await asyncio.to_thread(self._build_service)
        except Exception as e:
            logger.error(f"youtube: auth failed: {e}")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(out_queue), name="youtube-chat")
        logger.info("youtube: chat monitor started")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    def _build_service(self) -> None:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
        token_path = Path("scripts/token.json")
        creds: Optional[Credentials] = None
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), scopes)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self._client_secret_file, scopes)
                creds = flow.run_local_server(port=0)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")
        self._service = build("youtube", "v3", credentials=creds, cache_discovery=False)

    async def _run(self, out: asyncio.Queue[ChatMessage]) -> None:
        next_page: Optional[str] = None
        poll_interval = 2.0
        while not self._stop_event.is_set():
            try:
                resp = await asyncio.to_thread(
                    lambda: self._service.liveChatMessages()
                    .list(
                        liveChatId=self._live_chat_id,
                        part="id,snippet,authorDetails",
                        pageToken=next_page,
                    )
                    .execute()
                )
                poll_interval = max(1.0, resp.get("pollingIntervalMillis", 2000) / 1000.0)
                next_page = resp.get("nextPageToken")
                for item in resp.get("items", []):
                    snip = item.get("snippet", {})
                    author = item.get("authorDetails", {})
                    text = snip.get("displayMessage") or snip.get("textMessageDetails", {}).get("messageText", "")
                    is_super = snip.get("type", "").endswith("superChatEvent")
                    if text:
                        msg = ChatMessage(
                            platform="youtube",
                            username=author.get("displayName", "viewer"),
                            text=text,
                            is_highlight=bool(is_super),
                        )
                        try:
                            out.put_nowait(msg)
                        except asyncio.QueueFull:
                            pass
            except Exception as e:
                logger.warning(f"youtube: poll error: {e}")
                poll_interval = 5.0
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass
