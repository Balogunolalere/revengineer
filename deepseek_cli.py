#!/usr/bin/env python3
"""
DeepSeek Chat CLI — reverse-engineered terminal client for chat.deepseek.com

Interactive terminal chat with thinking mode, web search, and streaming.

=== QUICKSTART (Google login — recommended) ===

    # First time only: grab token via browser
    python3 grab_token.py
    # → Opens browser, log in with Google, token saved automatically

    # Then just run:
    python3 deepseek_cli.py
    # → Auto-loads token from ~/.deepseek_token

=== ALTERNATIVE: Manual token ===

    # Copy bearer token from browser DevTools → Network tab → Authorization header
    python3 deepseek_cli.py --token YOUR_BEARER_TOKEN

=== ALTERNATIVE: Email/password login ===

    # Only works if you registered with email, NOT Google login
    python3 deepseek_cli.py --email you@example.com --password yourpass

=== OPTIONS ===

    python3 deepseek_cli.py --no-think    # disable thinking mode (faster)
    python3 deepseek_cli.py --no-search   # disable web search

=== IN-CHAT COMMANDS ===

    /new     — start a new chat session
    /think   — toggle thinking mode on/off
    /search  — toggle web search on/off
    /quit    — exit

=== EXAMPLES ===

    $ python3 deepseek_cli.py
      Using saved token from ~/.deepseek_token
      Creating chat session... ✓ (a1b2c3d4...)
      Thinking: ON | Search: ON

    You: What is the mass of the sun?
    DeepSeek: The mass of the Sun is approximately 1.989 × 10³⁰ kg ...
"""

import argparse
import base64
import json
import os
import re
import struct
import subprocess
import sys
import time
import uuid
import array
import copy
import requests

# Default token file (shared with grab_token.py)
TOKEN_FILE = os.environ.get(
    "DEEPSEEK_TOKEN_FILE",
    os.path.expanduser("~/.deepseek_token"),
)


def load_saved_token() -> str:
    """Load token from file saved by grab_token.py."""
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    return ""

# ============================================================================
# Custom Keccak-256 (DeepSeekHashV1) — 23 rounds (skips round 0)
# Ported from DeepSeek's JS worker. NOT standard SHA3-256.
# ============================================================================

RC32 = array.array('I', [
    0x00000000, 0x00000001, 0x00000000, 0x00008082,
    0x80000000, 0x0000808A, 0x80000000, 0x80008000,
    0x00000000, 0x0000808B, 0x00000000, 0x80000001,
    0x80000000, 0x80008081, 0x80000000, 0x00008009,
    0x00000000, 0x0000008A, 0x00000000, 0x00000088,
    0x00000000, 0x80008009, 0x00000000, 0x8000000A,
    0x00000000, 0x8000808B, 0x80000000, 0x0000008B,
    0x80000000, 0x00008089, 0x80000000, 0x00008003,
    0x80000000, 0x00008002, 0x80000000, 0x00000080,
    0x00000000, 0x0000800A, 0x80000000, 0x8000000A,
    0x80000000, 0x80008081, 0x80000000, 0x00008080,
    0x00000000, 0x80000001, 0x80000000, 0x80008008,
])

V = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]
W_ROT = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]

MASK32 = 0xFFFFFFFF


def _i32(x):
    """Simulate JavaScript's 32-bit signed integer behavior."""
    x = x & MASK32
    if x >= 0x80000000:
        return x - 0x100000000
    return x


def _u32(x):
    """Unsigned 32-bit."""
    return x & MASK32


def absorb32(queue: bytearray, state: list):
    for r in range(0, len(queue), 8):
        n = r >> 2
        state[n] ^= _u32(
            (queue[r + 7] << 24) | (queue[r + 6] << 16) |
            (queue[r + 5] << 8) | queue[r + 4]
        )
        state[n + 1] ^= _u32(
            (queue[r + 3] << 24) | (queue[r + 2] << 16) |
            (queue[r + 1] << 8) | queue[r]
        )


def squeeze32(state: list, buf: bytearray):
    for r in range(0, len(buf), 8):
        n = r >> 2
        v1 = state[n + 1]
        v0 = state[n]
        buf[r] = v1 & 0xFF
        buf[r + 1] = (v1 >> 8) & 0xFF
        buf[r + 2] = (v1 >> 16) & 0xFF
        buf[r + 3] = (v1 >> 24) & 0xFF
        buf[r + 4] = v0 & 0xFF
        buf[r + 5] = (v0 >> 8) & 0xFF
        buf[r + 6] = (v0 >> 16) & 0xFF
        buf[r + 7] = (v0 >> 24) & 0xFF


def keccak_f(A: list):
    """Keccak-f[1600] permutation — 23 rounds (rounds 1-23, skipping round 0)."""
    C = [0] * 10

    for ri in range(1, 24):
        # theta
        for t in range(5):
            n2 = 2 * t
            C[n2] = _u32(A[n2] ^ A[n2 + 10] ^ A[n2 + 20] ^ A[n2 + 30] ^ A[n2 + 40])
            C[n2 + 1] = _u32(A[n2 + 1] ^ A[n2 + 11] ^ A[n2 + 21] ^ A[n2 + 31] ^ A[n2 + 41])

        for t in range(5):
            ci = ((t + 1) % 5) * 2
            o = C[ci]
            f = C[ci + 1]
            d0 = _u32(C[((t + 4) % 5) * 2] ^ _u32((o << 1) | (f >> 31)))
            d1 = _u32(C[((t + 4) % 5) * 2 + 1] ^ _u32((f << 1) | (o >> 31)))
            for r_idx in range(0, 25, 5):
                idx = (r_idx + t) * 2
                A[idx] = _u32(A[idx] ^ d0)
                A[idx + 1] = _u32(A[idx + 1] ^ d1)

        # rho + pi
        w0 = A[2]
        w1 = A[3]
        for ii in range(24):
            t_idx = V[ii]
            a_val = W_ROT[ii]
            c0 = A[2 * t_idx]
            c1 = A[2 * t_idx + 1]
            a_mod = a_val & 31
            s_mod = (32 - a_val) & 31
            v0 = _u32((w0 << a_mod) | (w1 >> s_mod))
            v1 = _u32((w1 << a_mod) | (w0 >> s_mod))
            if a_val < 32:
                w0, w1 = v0, v1
            else:
                w0, w1 = v1, v0
            A[2 * t_idx] = w0
            A[2 * t_idx + 1] = w1
            w0, w1 = c0, c1

        # chi
        for t in range(0, 25, 5):
            for n in range(5):
                C[2 * n] = A[(t + n) * 2]
                C[2 * n + 1] = A[(t + n) * 2 + 1]
            for n in range(5):
                idx2 = (t + n) * 2
                A[idx2] = _u32(A[idx2] ^ (_u32(~C[((n + 1) % 5) * 2]) & C[((n + 2) % 5) * 2]))
                A[idx2 + 1] = _u32(A[idx2 + 1] ^ (_u32(~C[((n + 1) % 5) * 2 + 1]) & C[((n + 2) % 5) * 2 + 1]))

        # iota
        n2 = 2 * ri
        A[0] = _u32(A[0] ^ RC32[n2])
        A[1] = _u32(A[1] ^ RC32[n2 + 1])


class DSKeccak:
    """DeepSeek's custom Keccak-256 hasher (23 rounds)."""

    def __init__(self):
        self.state = [0] * 50
        self.queue = bytearray(136)
        self.qoff = 0

    def update(self, s: str) -> 'DSKeccak':
        data = s.encode('utf-8')
        for byte in data:
            self.queue[self.qoff] = byte
            self.qoff += 1
            if self.qoff >= 136:
                absorb32(self.queue, self.state)
                keccak_f(self.state)
                self.qoff = 0
        return self

    def digest(self) -> str:
        st = list(self.state)
        q = bytearray(136)
        q[:len(self.queue)] = self.queue
        for i in range(self.qoff, 136):
            q[i] = 0
        q[self.qoff] |= 6
        q[135] |= 0x80
        absorb32(q, st)
        keccak_f(st)
        buf = bytearray(32)
        squeeze32(st, buf)
        return buf.hex()

    def copy(self) -> 'DSKeccak':
        k = DSKeccak()
        k.state = list(self.state)
        k.queue = bytearray(self.queue)
        k.qoff = self.qoff
        return k


# ============================================================================
# PoW Solver
# ============================================================================

def solve_pow(challenge_data: dict) -> str:
    """Solve DeepSeek's Proof of Work challenge and return base64-encoded response."""
    algorithm = challenge_data['algorithm']
    challenge = challenge_data['challenge']
    salt = challenge_data['salt']
    difficulty = int(challenge_data['difficulty'])
    expire_at = challenge_data['expire_at']
    signature = challenge_data['signature']
    target_path = challenge_data['target_path']

    if algorithm != 'DeepSeekHashV1':
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    print(f"  Solving PoW (difficulty={difficulty})...", end="", flush=True)
    start = time.time()

    # Try Node.js solver first (100x faster than Python)
    solver_js = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pow_solver.js")
    answer = None

    if os.path.exists(solver_js):
        try:
            result = subprocess.run(
                ["node", solver_js, challenge, salt, str(difficulty), str(expire_at)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if "answer" in data:
                    answer = data["answer"]
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass

    # Fallback to pure Python (slow but no dependencies)
    if answer is None:
        print(" (using Python fallback — this will be slow)...", end="", flush=True)
        prefix = f"{salt}_{expire_at}_"
        base = DSKeccak()
        base.update(prefix)

        for i in range(difficulty * 2):
            h = base.copy().update(str(i)).digest()
            if h == challenge:
                answer = i
                break
            if i % 10000 == 0 and i > 0:
                print(f"\r  Solving PoW... {i}/{difficulty} ({i*100//difficulty}%)",
                      end="", flush=True)

    elapsed = time.time() - start

    if answer is None:
        raise RuntimeError(f"PoW not solved after {difficulty*2} iterations")

    print(f"\r  PoW solved: answer={answer} in {elapsed:.1f}s       ")

    # Build the response payload
    pow_response = {
        "algorithm": algorithm,
        "challenge": challenge,
        "salt": salt,
        "answer": answer,
        "signature": signature,
        "target_path": target_path,
    }
    return base64.b64encode(json.dumps(pow_response).encode()).decode()


# ============================================================================
# DeepSeek API Client
# ============================================================================

BASE_URL = "https://chat.deepseek.com/api/v0"

COMMON_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-GB,en;q=0.9,en-US;q=0.8",
    "content-type": "application/json",
    "origin": "https://chat.deepseek.com",
    "sec-ch-ua": '"Not:A-Brand";v="99", "Microsoft Edge";v="145", "Chromium";v="145"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0",
    "x-app-version": "20241129.1",
    "x-client-locale": "en_US",
    "x-client-platform": "web",
    "x-client-timezone-offset": "3600",
    "x-client-version": "1.7.0",
}


class DeepSeekChat:
    def __init__(self, bearer_token: str = "", email: str = "", password: str = ""):
        self.token = bearer_token
        self.email = email
        self.password = password
        self.session = requests.Session()
        self.session.headers.update(COMMON_HEADERS)
        if bearer_token:
            self.session.headers["authorization"] = f"Bearer {bearer_token}"
        self.chat_session_id = None
        self.parent_message_id = None

    def login(self) -> str:
        """Login with email/password to get a fresh bearer token."""
        if not self.email or not self.password:
            raise RuntimeError("No email/password configured for login")

        payload = {
            "email": self.email,
            "password": self.password,
            "device_id": "deepseek_to_api",
            "os": "android",
        }
        resp = self.session.post(f"{BASE_URL}/users/login", json=payload)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 0:
            raise RuntimeError(f"Login failed: {result.get('msg', 'unknown')}")

        token = result["data"]["biz_data"]["user"]["token"]
        if not token:
            raise RuntimeError("Login returned empty token")

        self.token = token
        self.session.headers["authorization"] = f"Bearer {token}"
        return token

    @property
    def can_auto_login(self) -> bool:
        return bool(self.email and self.password)

    def _post(self, path: str, data: dict, extra_headers: dict = None) -> requests.Response:
        headers = extra_headers or {}
        return self.session.post(f"{BASE_URL}{path}", json=data, headers=headers)

    def create_session(self) -> str:
        """Create a new chat session."""
        resp = self._post("/chat_session/create", {})
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") != 0:
            raise RuntimeError(f"Failed to create session: {result}")
        self.chat_session_id = result["data"]["biz_data"]["id"]
        self.parent_message_id = None
        return self.chat_session_id

    def get_pow_challenge(self) -> dict:
        """Request a Proof of Work challenge."""
        resp = self._post("/chat/create_pow_challenge", {
            "target_path": "/api/v0/chat/completion"
        })
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") != 0:
            raise RuntimeError(f"Failed to get PoW challenge: {result}")
        return result["data"]["biz_data"]["challenge"]

    def send_message(self, prompt: str, thinking: bool = True, search: bool = True) -> str:
        """Send a message and stream the response. Returns the full response text."""
        # Step 1: Get PoW challenge
        print("  Getting PoW challenge...")
        challenge = self.get_pow_challenge()

        # Step 2: Solve PoW
        pow_response = solve_pow(challenge)

        # Step 3: Send completion request
        payload = {
            "chat_session_id": self.chat_session_id,
            "parent_message_id": self.parent_message_id,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking,
            "search_enabled": search,
            "preempt": False,
        }

        resp = self.session.post(
            f"{BASE_URL}/chat/completion",
            json=payload,
            headers={"x-ds-pow-response": pow_response},
            stream=True,
        )
        resp.raise_for_status()

        # Step 4: Parse SSE stream (DeepSeek uses JSON-patch format)
        full_content = []
        thinking_content = []
        is_thinking = False
        response_msg_id = None
        current_event = None
        current_frag_type = None  # "THINK", "RESPONSE", or "SEARCH"
        search_sources = {}  # cite_index → {url, title, site_name}

        CITATION_RE = re.compile(r'\[citation:(\d+)\]')

        def _format_citation(text: str) -> str:
            """Replace [citation:N] with [N] for cleaner terminal output."""
            return CITATION_RE.sub(r'[\1]', text)

        def _print_text(text: str, is_think: bool = False):
            """Print text, replacing citation tags inline."""
            display = _format_citation(text)
            if is_think:
                print(f"\033[2m{display}\033[0m", end="", flush=True)
            else:
                print(display, end="", flush=True)

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue

            # Track SSE event types
            if line.startswith("event: "):
                current_event = line[7:].strip()
                continue

            if not line.startswith("data: "):
                continue

            data_str = line[6:]

            # Handle special events
            if current_event == "ready":
                try:
                    ready = json.loads(data_str)
                    response_msg_id = ready.get("response_message_id")
                except json.JSONDecodeError:
                    pass
                current_event = None
                continue

            if current_event in ("close", "title", "update_session"):
                current_event = None
                continue

            current_event = None

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Handle full initial response object
            if "v" in event and isinstance(event["v"], dict) and "response" in event["v"]:
                resp_data = event["v"]["response"]
                fragments = resp_data.get("fragments", [])
                for frag in fragments:
                    frag_type = frag.get("type", "")
                    content = frag.get("content", "")
                    if frag_type == "SEARCH":
                        current_frag_type = "SEARCH"
                        # Extract search results if already present
                        for result in frag.get("results", []):
                            idx = result.get("cite_index")
                            if idx is not None:
                                search_sources[idx] = {
                                    "url": result.get("url", ""),
                                    "title": result.get("title", ""),
                                    "site_name": result.get("site_name", ""),
                                }
                        queries = frag.get("queries", [])
                        if queries:
                            print(f"\033[2m🔍 Searching: {', '.join(q.get('query','') for q in queries)}\033[0m")
                    elif frag_type == "THINK":
                        current_frag_type = "THINK"
                        is_thinking = True
                        print("\033[2m💭 ", end="", flush=True)
                        if content:
                            thinking_content.append(content)
                            _print_text(content, is_think=True)
                    elif frag_type == "RESPONSE":
                        if is_thinking:
                            is_thinking = False
                            print("\n\033[2m--- end thinking ---\033[0m\n", flush=True)
                        current_frag_type = "RESPONSE"
                        if content:
                            full_content.append(content)
                            _print_text(content)
                continue

            path = event.get("p", "")
            op = event.get("o", "")
            value = event.get("v", "")

            # Search results delivered (citation sources with URLs)
            if "results" in path and isinstance(value, list):
                for result in value:
                    if isinstance(result, dict) and "cite_index" in result:
                        search_sources[result["cite_index"]] = {
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "site_name": result.get("site_name", ""),
                        }
                continue

            # New fragment appended (thinking → response transition)
            if "fragments" in path and op == "APPEND" and isinstance(value, list):
                for frag in value:
                    frag_type = frag.get("type", "")
                    content = frag.get("content", "")
                    if frag_type == "SEARCH":
                        current_frag_type = "SEARCH"
                        for result in frag.get("results", []):
                            idx = result.get("cite_index")
                            if idx is not None:
                                search_sources[idx] = {
                                    "url": result.get("url", ""),
                                    "title": result.get("title", ""),
                                    "site_name": result.get("site_name", ""),
                                }
                    elif frag_type == "THINK":
                        current_frag_type = "THINK"
                        if not is_thinking:
                            is_thinking = True
                            print("\033[2m💭 ", end="", flush=True)
                        if content:
                            thinking_content.append(content)
                            _print_text(content, is_think=True)
                    elif frag_type == "RESPONSE":
                        if is_thinking:
                            is_thinking = False
                            print("\n\033[2m--- end thinking ---\033[0m\n", flush=True)
                        current_frag_type = "RESPONSE"
                        if content:
                            full_content.append(content)
                            _print_text(content)
                continue

            # Content append to current fragment
            if "content" in path and (op == "APPEND" or not op) and isinstance(value, str):
                if current_frag_type == "THINK":
                    thinking_content.append(value)
                    _print_text(value, is_think=True)
                else:
                    full_content.append(value)
                    _print_text(value)
                continue

            # Short-form: just {"v": "text"} — appends to current fragment
            if isinstance(value, str) and not path and not op:
                if current_frag_type == "THINK":
                    thinking_content.append(value)
                    _print_text(value, is_think=True)
                else:
                    full_content.append(value)
                    _print_text(value)
                continue

        print()  # newline after streaming

        # Print search sources if any were collected
        if search_sources:
            print("\n\033[1;36m📚 Sources:\033[0m")
            for idx in sorted(search_sources.keys()):
                src = search_sources[idx]
                site = f" ({src['site_name']})" if src.get("site_name") else ""
                print(f"  \033[1m[{idx}]\033[0m {src.get('title', 'Untitled')}{site}")
                print(f"      \033[4;34m{src['url']}\033[0m")

        if response_msg_id:
            self.parent_message_id = response_msg_id

        return "".join(full_content)


# ============================================================================
# Interactive CLI
# ============================================================================

def print_banner():
    print("""
\033[1;34m╔══════════════════════════════════════════╗
║   DeepSeek Chat CLI (reverse-engineered)  ║
╚══════════════════════════════════════════╝\033[0m
  Commands:
    /new     — start a new chat session
    /think   — toggle thinking mode
    /search  — toggle search mode
    /quit    — exit
""")


def main():
    parser = argparse.ArgumentParser(description="DeepSeek Chat CLI")
    parser.add_argument("--token", default="", help="Bearer token from browser DevTools")
    parser.add_argument("--email", default="", help="DeepSeek account email (auto-login)")
    parser.add_argument("--password", default="", help="DeepSeek account password (auto-login)")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode")
    parser.add_argument("--no-search", action="store_true", help="Disable web search")
    args = parser.parse_args()

    if args.email and args.password:
        client = DeepSeekChat(email=args.email, password=args.password)
        print("  Logging in...", end="", flush=True)
        try:
            client.login()
            print(" ✓")
        except Exception as e:
            print(f"\n\033[31mLogin failed: {e}\033[0m")
            sys.exit(1)
    elif args.token:
        client = DeepSeekChat(bearer_token=args.token)
    else:
        # Try auto-loading from saved token file (grab_token.py)
        saved = load_saved_token()
        if saved:
            print(f"  Using saved token from {TOKEN_FILE}")
            client = DeepSeekChat(bearer_token=saved)
        else:
            print("Error: provide --email + --password, --token, or run grab_token.py first")
            sys.exit(1)

    thinking = not args.no_think
    search = not args.no_search

    print_banner()

    # Create initial session
    print("  Creating chat session...", end="", flush=True)
    session_id = client.create_session()
    print(f" ✓ ({session_id[:8]}...)")
    print(f"  Thinking: {'ON' if thinking else 'OFF'} | Search: {'ON' if search else 'OFF'}")
    print()

    while True:
        try:
            prompt = input("\033[1;32mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        if prompt.lower() == "/quit":
            print("Bye!")
            break
        elif prompt.lower() == "/new":
            print("  Creating new session...", end="", flush=True)
            session_id = client.create_session()
            print(f" ✓ ({session_id[:8]}...)")
            continue
        elif prompt.lower() == "/think":
            thinking = not thinking
            print(f"  Thinking: {'ON' if thinking else 'OFF'}")
            continue
        elif prompt.lower() == "/search":
            search = not search
            print(f"  Search: {'ON' if search else 'OFF'}")
            continue

        print(f"\033[1;35mDeepSeek:\033[0m", flush=True)
        try:
            client.send_message(prompt, thinking=thinking, search=search)
        except Exception as e:
            print(f"\n\033[31mError: {e}\033[0m")
        print()


if __name__ == "__main__":
    main()
