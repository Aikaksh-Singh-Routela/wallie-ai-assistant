"""YouTube live chat monitor via Data API v3."""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Optional

from loguru import logger

from .base import ChatMessage, ChatMonitor


def _video_id(s: str) -> str:
    """Extract an 11-char YouTube video id from a URL or bare id; else ''."""
    s = (s or "").strip()
    m = re.search(r"(?:v=|youtu\.be/|/live/|/watch\?v=|/shorts/)([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    return ""


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
        try:
            await asyncio.to_thread(self._build_service)
        except Exception as e:
            logger.error(f"youtube: auth failed: {e}")
            return
        try:
            self._live_chat_id = await asyncio.to_thread(self._resolve_chat_id, self._live_chat_id)
        except Exception as e:
            logger.error(f"youtube: could not resolve live chat id: {e}")
            return
        if not self._live_chat_id:
            logger.warning(
                "youtube: no active live chat found — go live, or set YOUTUBE_LIVE_CHAT_ID "
                "to your stream URL/video id; monitor disabled"
            )
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(out_queue), name="youtube-chat")
        logger.info(f"youtube: chat monitor started (liveChatId={self._live_chat_id[:18]}…)")

    def _resolve_chat_id(self, ref: str) -> str:
        """Turn whatever's configured into a concrete liveChatId.

        - empty            → your own currently-active broadcast
        - a video URL / id → that live stream's active chat
        - a raw liveChatId → used as-is
        """
        ref = (ref or "").strip()
        if ref and "youtu" not in ref and not _video_id(ref) and len(ref) > 20:
            return ref  # already a liveChatId
        if ref:
            vid = _video_id(ref)
            if not vid:
                logger.warning(f"youtube: '{ref[:40]}' isn't a video URL/id or a chat id")
                return ""
            r = self._service.videos().list(part="liveStreamingDetails", id=vid).execute()
            items = r.get("items", [])
            cid = (items[0].get("liveStreamingDetails", {}) or {}).get("activeLiveChatId") if items else ""
            if not cid:
                logger.warning("youtube: that video isn't live or has chat disabled")
            return cid or ""
        # empty → resolve your own active broadcast
        r = self._service.liveBroadcasts().list(
            part="snippet", broadcastStatus="active", mine=True
        ).execute()
        items = r.get("items", [])
        if not items:
            return ""
        return items[0]["snippet"].get("liveChatId", "")

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

    _HIGHLIGHT_TYPES = {
        "superChatEvent", "superStickerEvent",
        "newSponsorEvent", "memberMilestoneChatEvent",
    }

    async def _run(self, out: asyncio.Queue[ChatMessage]) -> None:
        next_page: Optional[str] = None
        poll_interval = 2.0
        warmed_up = False
        while not self._stop_event.is_set():
            try:
                page_token = next_page
                resp = await asyncio.to_thread(
                    lambda pt=page_token: self._service.liveChatMessages()
                    .list(
                        liveChatId=self._live_chat_id,
                        part="id,snippet,authorDetails",
                        pageToken=pt,
                    )
                    .execute()
                )
                poll_interval = max(1.0, resp.get("pollingIntervalMillis", 2000) / 1000.0)
                next_page = resp.get("nextPageToken")
                if not warmed_up:
                    warmed_up = True
                    logger.debug(f"youtube: skipped {len(resp.get('items', []))} backlog messages")
                    continue
                for item in resp.get("items", []):
                    snip = item.get("snippet", {})
                    author = item.get("authorDetails", {})
                    text = snip.get("displayMessage") or snip.get("textMessageDetails", {}).get("messageText", "")
                    msg_type = snip.get("type", "")
                    is_highlight = msg_type in self._HIGHLIGHT_TYPES
                    if text:
                        try:
                            out.put_nowait(ChatMessage(
                                platform="youtube",
                                username=author.get("displayName", "viewer"),
                                text=text,
                                is_highlight=is_highlight,
                            ))
                        except asyncio.QueueFull:
                            pass
            except Exception as e:
                err_str = str(e).lower()
                if "401" in err_str or "403" in err_str or "unauthorized" in err_str:
                    logger.warning("youtube: auth expired, refreshing credentials")
                    try:
                        await asyncio.to_thread(self._build_service)
                        warmed_up = False
                        next_page = None
                    except Exception as auth_err:
                        logger.error(f"youtube: re-auth failed: {auth_err}")
                else:
                    logger.warning(f"youtube: poll error: {e}")
                poll_interval = 5.0
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass
