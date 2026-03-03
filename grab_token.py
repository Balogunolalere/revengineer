#!/usr/bin/env python3
"""
DeepSeek Token Grabber — opens a real browser for Google OAuth login,
intercepts the bearer token from API calls, and saves it for the CLI/API.

Since DeepSeek supports Google OAuth login, you can't use email/password to
get a token automatically. This script opens a real browser, you log in with
Google (or email), and it captures the bearer token from network requests.

The token is saved to ~/.deepseek_token and all other tools auto-load it.

=== SETUP (one time) ===

    pip install playwright requests
    playwright install chromium

=== STEP 1: GRAB YOUR TOKEN ===

    python3 grab_token.py

    This opens a Chromium browser window. Log in with Google (or email/password).
    Once you're logged in and see the DeepSeek chat page, the token is captured
    automatically. The browser closes and the token is saved to ~/.deepseek_token.

    Your Google login cookies are saved in ~/.deepseek_browser_profile so next
    time you run this, Google may auto-login without asking for credentials.

=== STEP 2: USE THE CLI OR API (token loads automatically) ===

    # Interactive chat in terminal:
    python3 deepseek_cli.py

    # Start OpenAI-compatible API server:
    python3 deepseek_api.py

    # Both auto-load token from ~/.deepseek_token — no args needed!

=== OTHER COMMANDS ===

    python3 grab_token.py --validate   # check if saved token still works
    python3 grab_token.py --show       # print the raw token string
    python3 grab_token.py --force      # force re-login even if token is valid

=== WHEN TOKEN EXPIRES ===

    Just run `python3 grab_token.py` again. Google cookies are remembered
    so it should auto-login without re-entering your password.
"""

import argparse
import json
import os
import sys
import time

import requests

TOKEN_FILE = os.environ.get(
    "DEEPSEEK_TOKEN_FILE",
    os.path.expanduser("~/.deepseek_token"),
)

DEEPSEEK_BASE = "https://chat.deepseek.com"
VALIDATE_URL = f"{DEEPSEEK_BASE}/api/v0/users/current"

HEADERS = {
    "accept": "*/*",
    "content-type": "application/json",
    "origin": DEEPSEEK_BASE,
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    ),
    "x-app-version": "20241129.1",
    "x-client-platform": "web",
    "x-client-version": "1.7.0",
}

# Browser profile directory — persists Google login cookies so you
# don't have to re-enter your Google credentials every time.
BROWSER_PROFILE = os.path.expanduser("~/.deepseek_browser_profile")


def save_token(token: str) -> None:
    """Save token to file."""
    with open(TOKEN_FILE, "w") as f:
        f.write(token.strip())
    os.chmod(TOKEN_FILE, 0o600)
    print(f"  Token saved to {TOKEN_FILE}")


def load_token() -> str:
    """Load token from file."""
    if not os.path.exists(TOKEN_FILE):
        return ""
    with open(TOKEN_FILE) as f:
        return f.read().strip()


def validate_token(token: str) -> dict | None:
    """Validate a token against DeepSeek API. Returns user info or None."""
    if not token:
        return None
    try:
        resp = requests.get(
            VALIDATE_URL,
            headers={**HEADERS, "authorization": f"Bearer {token}"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("code") != 0:
            return None
        biz_data = data.get("data", {}).get("biz_data", {})
        if biz_data:
            return biz_data
        return None
    except Exception:
        return None


def grab_token_via_browser() -> str:
    """Open a real browser, wait for user to login, capture the token."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Error: playwright not installed.")
        print("  pip install playwright && playwright install chromium")
        sys.exit(1)

    captured_token = {"value": None}

    def on_request(request):
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer ") and "chat.deepseek.com/api/" in request.url:
            token = auth[7:].strip()
            if token and len(token) > 20:
                captured_token["value"] = token

    print(f"\n  Opening browser... (profile: {BROWSER_PROFILE})")
    print("  Log in with Google, then the token will be captured automatically.\n")

    with sync_playwright() as p:
        # Use persistent context so Google login cookies are remembered
        context = p.chromium.launch_persistent_context(
            user_data_dir=BROWSER_PROFILE,
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
            viewport={"width": 1280, "height": 800},
        )

        page = context.pages[0] if context.pages else context.new_page()
        page.on("request", on_request)

        # Navigate to DeepSeek
        page.goto(DEEPSEEK_BASE, wait_until="domcontentloaded")

        print("  Waiting for you to log in...")
        print("  (The browser will close automatically once a token is captured)\n")

        # Poll for token capture
        max_wait = 300  # 5 minutes
        start = time.time()
        while time.time() - start < max_wait:
            if captured_token["value"]:
                # Validate it
                user = validate_token(captured_token["value"])
                if user:
                    email = user.get("email", "unknown")
                    name = user.get("name", "")
                    print(f"  Token captured! Logged in as: {email}" +
                          (f" ({name})" if name else ""))
                    break
                else:
                    # Got a token but it didn't validate — keep waiting
                    captured_token["value"] = None

            # Check if page navigated to chat (means login succeeded)
            try:
                page.wait_for_timeout(1000)
            except Exception:
                break

            # Check if browser was closed by user
            if not context.pages:
                break

        context.close()

    token = captured_token["value"]
    if not token:
        print("\n  No token captured. Did you complete the login?")
        sys.exit(1)

    return token


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek token grabber — browser-based login for Google OAuth"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Check if saved token is still valid"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Print the current saved token"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-login even if current token is valid"
    )
    args = parser.parse_args()

    # --show: just print token
    if args.show:
        token = load_token()
        if token:
            print(token)
        else:
            print("No saved token found.", file=sys.stderr)
            sys.exit(1)
        return

    # --validate: check existing token
    if args.validate:
        token = load_token()
        if not token:
            print("No saved token found.")
            sys.exit(1)
        user = validate_token(token)
        if user:
            email = user.get("email", "unknown")
            name = user.get("name", "")
            print(f"Token is valid — {email}" + (f" ({name})" if name else ""))
        else:
            print("Token is expired or invalid.")
            sys.exit(1)
        return

    # Default: grab a new token
    existing = load_token()
    if existing and not args.force:
        user = validate_token(existing)
        if user:
            email = user.get("email", "unknown")
            print(f"  Existing token is still valid ({email}). Use --force to re-login.")
            return

    token = grab_token_via_browser()
    save_token(token)
    print("\n  Done! You can now use:")
    print(f"    python3 deepseek_cli.py          # auto-loads token from {TOKEN_FILE}")
    print(f"    python3 deepseek_api.py           # auto-loads token from {TOKEN_FILE}")


if __name__ == "__main__":
    main()
