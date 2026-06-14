"""Standalone YouTube live-chat reader — test Wallie's chat reading WITHOUT going live.

  python scripts/test_youtube_chat.py <youtube live url or video id>   # read ANY live stream
  python scripts/test_youtube_chat.py                                  # read YOUR OWN active broadcast

First run opens a browser for a one-time Google sign-in and saves scripts/token.json
(the same token the live app reuses). Point it at any currently-live YouTube stream to
prove chat reading works before you ever go live. Ctrl+C to stop.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

try:  # chat messages are full of emojis; keep the Windows console from crashing
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]
HERE = Path(__file__).resolve().parent
CLIENT_SECRET = HERE / "client_secret.json"
TOKEN = HERE / "token.json"


def auth():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CLIENT_SECRET.exists():
                sys.exit(f"client_secret.json bulunamadi: {CLIENT_SECRET}")
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)
            print(">>> Asagidaki URL'yi BRAVE'de (yelkbaba17 hesabinin oldugu tarayici) ac, giris yap, izin ver:\n")
            creds = flow.run_local_server(port=0, open_browser=False)
        TOKEN.write_text(creds.to_json(), encoding="utf-8")
        print(f"OK token kaydedildi -> {TOKEN}")
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _video_id(s: str) -> str:
    s = s.strip()
    m = re.search(r"(?:v=|youtu\.be/|/live/|/watch\?v=|/shorts/)([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    return ""


def resolve_chat_id(svc, arg: str) -> str:
    if arg:
        vid = _video_id(arg)
        if not vid:
            sys.exit(f"Gecerli bir YouTube video URL/ID degil: {arg}")
        r = svc.videos().list(part="liveStreamingDetails,snippet", id=vid).execute()
        items = r.get("items", [])
        if not items:
            sys.exit("Video bulunamadi.")
        cid = (items[0].get("liveStreamingDetails", {}) or {}).get("activeLiveChatId")
        title = items[0].get("snippet", {}).get("title", "")
        if not cid:
            sys.exit("Bu video su an CANLI degil ya da chat kapali. Aktif bir canli yayin linki ver.")
        print(f"Yayin: {title}")
        return cid
    # no arg -> your own active broadcast
    r = svc.liveBroadcasts().list(part="snippet", broadcastStatus="active", mine=True).execute()
    items = r.get("items", [])
    if not items:
        sys.exit(
            "Senin aktif canli yayinin yok.\n"
            "Yayin acmadan test icin herhangi bir canli yayin linki ver:\n"
            "  python scripts/test_youtube_chat.py https://www.youtube.com/watch?v=XXXXXXXXXXX"
        )
    cid = items[0]["snippet"]["liveChatId"]
    print(f"Kendi yayinin: {items[0]['snippet'].get('title', '')}")
    return cid


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    svc = auth()
    chat_id = resolve_chat_id(svc, arg)
    print(f"liveChatId: {chat_id}\n--- Chat dinleniyor (Ctrl+C ile cik) ---\n")
    page = None
    warmed = False
    while True:
        resp = svc.liveChatMessages().list(
            liveChatId=chat_id, part="snippet,authorDetails", pageToken=page
        ).execute()
        page = resp.get("nextPageToken")
        wait = max(1.0, resp.get("pollingIntervalMillis", 2000) / 1000.0)
        items = resp.get("items", [])
        if not warmed:  # skip the backlog on first poll, like the live app does
            warmed = True
            print(f"(baslangic: {len(items)} eski mesaj atlandi — yeni mesajlari bekliyorum)\n")
        else:
            for it in items:
                who = it.get("authorDetails", {}).get("displayName", "viewer")
                txt = it.get("snippet", {}).get("displayMessage", "")
                if txt:
                    print(f"  {who}: {txt}")
        time.sleep(wait)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbitti.")
