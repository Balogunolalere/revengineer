#!/usr/bin/env python3
"""
DeepSeek → OpenAI-compatible API Server (FastAPI)

Wraps the reverse-engineered chat.deepseek.com into an OpenAI-compatible
/v1/chat/completions endpoint. Works with any tool that supports OpenAI's API:
Continue, Cursor, aider, LangChain, OpenAI Python SDK, etc.

=== QUICKSTART (Google login — recommended) ===

    # First time: grab token via browser
    python3 grab_token.py

    # Start the API server (auto-loads token from ~/.deepseek_token):
    python3 deepseek_api.py

    # Server runs on http://localhost:8000

=== ALTERNATIVE: Manual token ===

    python3 deepseek_api.py --token YOUR_BEARER_TOKEN
    # or
    export DEEPSEEK_TOKEN="your_token"
    python3 deepseek_api.py

=== ALTERNATIVE: Email/password (NOT Google login) ===

    python3 deepseek_api.py --email you@example.com --password yourpass
    # Token auto-refreshes when it expires

=== MAKING REQUESTS ===

    # Basic chat:
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hello"}]}'

    # With thinking (chain-of-thought reasoning):
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "Solve x^2+3x+2=0"}]}'

    # With web search:
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "deepseek-search", "messages": [{"role": "user", "content": "Latest AI news"}]}'

    # Streaming:
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'

=== USING WITH PYTHON (OpenAI SDK) ===

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(resp.choices[0].message.content)

=== MODELS ===

    deepseek-chat       — default chat (no thinking, no search)
    deepseek-reasoner   — thinking/reasoning mode enabled (alias: deepseek-r1)
    deepseek-search     — web search enabled

=== ENDPOINTS ===

    POST /v1/chat/completions   — OpenAI-compatible chat completions
    GET  /v1/models             — list available models
    GET  /health                — health check
    GET  /stats                 — request stats (total, success, failed, token refreshes)

=== OPTIONS ===

    --port 8080     change port (default: 8000)
    --host 0.0.0.0  change bind address
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import time
import uuid
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Import PoW solver from the CLI module
from deepseek_cli import DSKeccak

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deepseek-api")

# ============================================================================
# Config
# ============================================================================
DEEPSEEK_BASE = "https://chat.deepseek.com/api/v0"

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
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
    ),
    "x-app-version": "20241129.1",
    "x-client-locale": "en_US",
    "x-client-platform": "web",
    "x-client-timezone-offset": "3600",
    "x-client-version": "1.7.0",
}


# ============================================================================
# PoW Solver (async, calls Node.js subprocess)
# ============================================================================
SOLVER_JS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pow_solver.js")


async def solve_pow_async(challenge_data: dict) -> str:
    """Solve PoW using Node.js subprocess (async)."""
    algorithm = challenge_data["algorithm"]
    challenge = challenge_data["challenge"]
    salt = challenge_data["salt"]
    difficulty = int(challenge_data["difficulty"])
    expire_at = challenge_data["expire_at"]
    signature = challenge_data["signature"]
    target_path = challenge_data["target_path"]

    if algorithm != "DeepSeekHashV1":
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    start = time.time()

    if os.path.exists(SOLVER_JS):
        proc = await asyncio.create_subprocess_exec(
            "node", SOLVER_JS, challenge, salt, str(difficulty), str(expire_at),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        if proc.returncode == 0:
            data = json.loads(stdout.decode())
            answer = data.get("answer")
        else:
            raise RuntimeError(f"PoW solver failed: {stderr.decode()}")
    else:
        # Python fallback (slow)
        prefix = f"{salt}_{expire_at}_"
        base = DSKeccak()
        base.update(prefix)
        answer = None

        for i in range(difficulty * 2):
            h = base.copy().update(str(i)).digest()
            if h == challenge:
                answer = i
                break

    elapsed = time.time() - start

    if answer is None:
        raise RuntimeError("PoW not solved")

    log.info(f"PoW solved: answer={answer} in {elapsed:.1f}s")

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
# DeepSeek Async Client
# ============================================================================
class DeepSeekClient:
    """Async client for chat.deepseek.com with auto-login support."""

    LOGIN_URL = f"{DEEPSEEK_BASE}/users/login"

    def __init__(self, bearer_token: str = "", email: str = "", password: str = ""):
        self.token = bearer_token
        self.email = email
        self.password = password
        self._client: Optional[httpx.AsyncClient] = None
        # Request tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.token_refreshes = 0
        self.start_time = time.time()

    def _build_client(self) -> httpx.AsyncClient:
        """Build httpx client with current token."""
        return httpx.AsyncClient(
            headers={**COMMON_HEADERS, "authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = self._build_client()
        return self._client

    async def login(self) -> str:
        """Login with email/password to get a fresh bearer token."""
        if not self.email or not self.password:
            raise RuntimeError("No email/password configured for auto-login")

        payload = {
            "email": self.email,
            "password": self.password,
            "device_id": "deepseek_to_api",
            "os": "android",
        }

        async with httpx.AsyncClient(
            headers=COMMON_HEADERS, timeout=httpx.Timeout(30.0)
        ) as tmp:
            resp = await tmp.post(self.LOGIN_URL, json=payload)
            resp.raise_for_status()
            result = resp.json()

        code = result.get("code", -1)
        if code != 0:
            raise RuntimeError(f"Login failed: {result.get('msg', 'unknown')}")

        biz_data = result["data"]["biz_data"]
        token = biz_data["user"]["token"]
        if not token:
            raise RuntimeError("Login returned empty token")

        self.token = token
        # Rebuild client with new token
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = self._build_client()
        self.token_refreshes += 1
        log.info(f"Login successful (refresh #{self.token_refreshes})")
        return token

    def _is_token_invalid(self, status_code: int, resp_body: dict) -> bool:
        """Check if the response indicates an expired/invalid token."""
        if status_code in (401, 403):
            return True
        code = resp_body.get("code", 0)
        if code in (40001, 40002, 40003):
            return True
        msg = str(resp_body.get("msg", "")).lower()
        return "token" in msg or "unauthorized" in msg

    @property
    def can_auto_login(self) -> bool:
        return bool(self.email and self.password)

    async def create_session(self) -> str:
        resp = await self.client.post(f"{DEEPSEEK_BASE}/chat_session/create", json={})
        result = resp.json()

        # Auto-refresh on token failure
        if self._is_token_invalid(resp.status_code, result) and self.can_auto_login:
            log.warning("Token invalid on create_session, re-logging in...")
            await self.login()
            resp = await self.client.post(f"{DEEPSEEK_BASE}/chat_session/create", json={})
            result = resp.json()

        resp.raise_for_status()
        if result.get("code") != 0:
            raise RuntimeError(f"Failed to create session: {result}")
        return result["data"]["biz_data"]["id"]

    async def get_pow_challenge(self) -> dict:
        resp = await self.client.post(
            f"{DEEPSEEK_BASE}/chat/create_pow_challenge",
            json={"target_path": "/api/v0/chat/completion"},
        )
        result = resp.json()

        # Auto-refresh on token failure
        if self._is_token_invalid(resp.status_code, result) and self.can_auto_login:
            log.warning("Token invalid on get_pow, re-logging in...")
            await self.login()
            resp = await self.client.post(
                f"{DEEPSEEK_BASE}/chat/create_pow_challenge",
                json={"target_path": "/api/v0/chat/completion"},
            )
            result = resp.json()

        resp.raise_for_status()
        if result.get("code") != 0:
            raise RuntimeError(f"Failed to get PoW: {result}")
        return result["data"]["biz_data"]["challenge"]

    async def _prepare_request(
        self, messages: list[dict], thinking: bool, search: bool
    ) -> tuple[dict, str, str]:
        """Create session, solve PoW, build payload. Returns (payload, pow_response, session_id)."""
        chat_session_id = await self.create_session()
        prompt = self._messages_to_prompt(messages)
        challenge = await self.get_pow_challenge()
        pow_response = await solve_pow_async(challenge)

        payload = {
            "chat_session_id": chat_session_id,
            "parent_message_id": None,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking,
            "search_enabled": search,
            "preempt": False,
        }
        return payload, pow_response, chat_session_id

    async def chat_completion(
        self,
        messages: list[dict],
        thinking: bool = False,
        search: bool = False,
        stream: bool = False,
        max_retries: int = 3,
    ) -> AsyncGenerator | dict:
        """Send a chat completion request. Returns async generator if stream=True.

        Retries with fresh session + PoW on upstream 5xx errors.
        """
        self.total_requests += 1

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                payload, pow_response, session_id = await self._prepare_request(
                    messages, thinking, search
                )
                if stream:
                    return self._stream_completion(payload, pow_response, session_id)
                else:
                    return await self._non_stream_completion(payload, pow_response, session_id)
            except HTTPException as e:
                last_error = e
                if e.status_code < 500 or attempt >= max_retries - 1:
                    raise
                delay = 2 ** attempt + 1
                log.warning(
                    f"Upstream {e.status_code} on attempt {attempt+1}/{max_retries}, "
                    f"retrying with fresh session in {delay}s"
                )
                await asyncio.sleep(delay)
            except Exception as e:
                last_error = e
                if attempt >= max_retries - 1:
                    raise
                delay = 2 ** attempt + 1
                log.warning(
                    f"Upstream error on attempt {attempt+1}/{max_retries}: {e}, "
                    f"retrying in {delay}s"
                )
                await asyncio.sleep(delay)

        raise last_error  # type: ignore[misc]

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert OpenAI messages format to a single prompt string."""
        if len(messages) == 1:
            return messages[0].get("content", "")

        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System Instructions]: {content}")
            elif role == "assistant":
                parts.append(f"[Previous Assistant Response]: {content}")
            elif role == "user":
                parts.append(content)
        return "\n\n".join(parts)

    async def _non_stream_completion(
        self, payload: dict, pow_response: str, session_id: str
    ) -> dict:
        """Non-streaming completion — collect full response."""
        full_content = []
        thinking_content = []
        search_sources = {}

        async for chunk_type, text in self._parse_sse_stream(payload, pow_response):
            if chunk_type == "think":
                thinking_content.append(text)
            elif chunk_type == "content":
                full_content.append(text)
            elif chunk_type == "sources":
                search_sources = json.loads(text)

        self.successful_requests += 1
        content = "".join(full_content)

        # Append search sources as markdown references
        if search_sources:
            refs = ["\n\n---\n**Sources:**"]
            for idx in sorted(int(k) for k in search_sources):
                src = search_sources[str(idx)] if str(idx) in search_sources else search_sources.get(idx, {})
                title = src.get("title", "Untitled")
                url = src.get("url", "")
                site = src.get("site_name", "")
                label = f"{title} ({site})" if site else title
                refs.append(f"[{idx}] [{label}]({url})")
            content += "\n".join(refs)

        req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }

    async def _stream_completion(
        self, payload: dict, pow_response: str, session_id: str
    ) -> AsyncGenerator[str, None]:
        """Streaming completion — yield SSE chunks in OpenAI format."""
        req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        search_sources = {}

        # Send initial role chunk
        initial = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(initial)}\n\n"

        async for chunk_type, text in self._parse_sse_stream(payload, pow_response):
            if chunk_type == "content" and text:
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "deepseek-chat",
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            elif chunk_type == "sources":
                search_sources = json.loads(text)

        # Append search sources as a final content chunk
        if search_sources:
            refs = ["\n\n---\n**Sources:**"]
            for idx in sorted(int(k) for k in search_sources):
                src = search_sources[str(idx)] if str(idx) in search_sources else search_sources.get(idx, {})
                title = src.get("title", "Untitled")
                url = src.get("url", "")
                site = src.get("site_name", "")
                label = f"{title} ({site})" if site else title
                refs.append(f"[{idx}] [{label}]({url})")
            sources_text = "\n".join(refs)
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "deepseek-chat",
                "choices": [{"index": 0, "delta": {"content": sources_text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Send final chunk
        final = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"
        self.successful_requests += 1

    async def _parse_sse_stream(
        self, payload: dict, pow_response: str
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Parse DeepSeek's SSE stream, yielding (type, text) tuples.
        Types: 'think', 'content', 'sources' (JSON-encoded search results)."""
        current_frag_type = None
        current_event = None
        search_sources = {}  # cite_index → {url, title, site_name}

        CITATION_RE = re.compile(r'\[citation:(\d+)\]')

        def _clean(text: str) -> str:
            return CITATION_RE.sub(r'[\1]', text)

        def _collect_results(results: list):
            for result in results:
                if isinstance(result, dict) and "cite_index" in result:
                    search_sources[result["cite_index"]] = {
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "site_name": result.get("site_name", ""),
                    }

        async with self.client.stream(
            "POST",
            f"{DEEPSEEK_BASE}/chat/completion",
            json=payload,
            headers={"x-ds-pow-response": pow_response},
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                log.error(f"Completion failed ({resp.status_code}): {body.decode()}")
                self.failed_requests += 1
                raise HTTPException(status_code=resp.status_code, detail="DeepSeek error")

            async for line in resp.aiter_lines():
                if not line:
                    continue

                if line.startswith("event: "):
                    current_event = line[7:].strip()
                    continue

                if not line.startswith("data: "):
                    continue

                data_str = line[6:]

                if current_event in ("ready", "close", "title", "update_session"):
                    current_event = None
                    continue
                current_event = None

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Initial response with fragments
                if "v" in event and isinstance(event["v"], dict) and "response" in event["v"]:
                    resp_data = event["v"]["response"]
                    for frag in resp_data.get("fragments", []):
                        frag_type = frag.get("type", "")
                        content = frag.get("content", "")
                        if frag_type == "SEARCH":
                            current_frag_type = "SEARCH"
                            _collect_results(frag.get("results", []))
                        elif frag_type == "THINK":
                            current_frag_type = "THINK"
                            if content:
                                yield ("think", _clean(content))
                        elif frag_type == "RESPONSE":
                            current_frag_type = "RESPONSE"
                            if content:
                                yield ("content", _clean(content))
                    continue

                path = event.get("p", "")
                op = event.get("o", "")
                value = event.get("v", "")

                # Search results delivered (citation sources with URLs)
                if "results" in path and isinstance(value, list):
                    _collect_results(value)
                    continue

                # New fragment
                if "fragments" in path and op == "APPEND" and isinstance(value, list):
                    for frag in value:
                        frag_type = frag.get("type", "")
                        content = frag.get("content", "")
                        if frag_type == "SEARCH":
                            current_frag_type = "SEARCH"
                            _collect_results(frag.get("results", []))
                        elif frag_type == "THINK":
                            current_frag_type = "THINK"
                            if content:
                                yield ("think", _clean(content))
                        elif frag_type == "RESPONSE":
                            current_frag_type = "RESPONSE"
                            if content:
                                yield ("content", _clean(content))
                    continue

                # Content append
                if "content" in path and isinstance(value, str):
                    if current_frag_type == "THINK":
                        yield ("think", _clean(value))
                    else:
                        yield ("content", _clean(value))
                    continue

                # Short-form value
                if isinstance(value, str) and not path and not op:
                    if current_frag_type == "THINK":
                        yield ("think", _clean(value))
                    else:
                        yield ("content", _clean(value))
                    continue

        # After stream ends, yield collected search sources
        if search_sources:
            yield ("sources", json.dumps(search_sources))

    def get_stats(self) -> dict:
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.total_requests,
            "successful": self.successful_requests,
            "failed": self.failed_requests,
            "token_refreshes": self.token_refreshes,
            "auth_mode": "email/password" if self.can_auto_login else "manual token",
            "requests_per_minute": round(self.successful_requests / max(uptime / 60, 0.01), 2),
        }


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(title="DeepSeek Proxy API", version="1.0.0")
client: Optional[DeepSeekClient] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0, le=2)
    max_tokens: Optional[int] = None


@app.on_event("startup")
async def startup():
    global client
    token = os.environ.get("DEEPSEEK_TOKEN", "")
    email = os.environ.get("DEEPSEEK_EMAIL", "")
    password = os.environ.get("DEEPSEEK_PASSWORD", "")

    if email and password:
        client = DeepSeekClient(email=email, password=password)
        try:
            await client.login()
            log.info("DeepSeek proxy ready (logged in with email/password)")
        except Exception as e:
            log.error(f"Login failed: {e}")
            return
    elif token:
        client = DeepSeekClient(bearer_token=token)
        log.info("DeepSeek proxy ready (using manual token)")
    else:
        # Try auto-loading from saved token file (grab_token.py)
        saved = load_saved_token()
        if saved:
            client = DeepSeekClient(bearer_token=saved)
            log.info(f"DeepSeek proxy ready (token from {TOKEN_FILE})")
        else:
            log.error("Set DEEPSEEK_EMAIL+DEEPSEEK_PASSWORD, DEEPSEEK_TOKEN, or run grab_token.py")
            return


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-chat",
                "object": "model",
                "created": 1700000000,
                "owned_by": "deepseek",
            },
            {
                "id": "deepseek-reasoner",
                "object": "model",
                "created": 1700000000,
                "owned_by": "deepseek",
            },
            {
                "id": "deepseek-search",
                "object": "model",
                "created": 1700000000,
                "owned_by": "deepseek",
            },
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not client:
        raise HTTPException(status_code=500, detail="DeepSeek not configured — set credentials")

    thinking = req.model in ("deepseek-reasoner", "deepseek-r1")
    search = req.model in ("deepseek-search",)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        if req.stream:
            generator = await client.chat_completion(
                messages=messages, thinking=thinking, search=search, stream=True
            )
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            result = await client.chat_completion(
                messages=messages, thinking=thinking, search=search, stream=False
            )
            return JSONResponse(result)
    except Exception as e:
        log.error(f"Request failed: {e}")
        client.failed_requests += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    if not client:
        return {"error": "not initialized"}
    return client.get_stats()


@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="DeepSeek OpenAI-compatible API proxy")
    parser.add_argument("--token", default=os.environ.get("DEEPSEEK_TOKEN", ""),
                        help="DeepSeek bearer token (manual)")
    parser.add_argument("--email", default=os.environ.get("DEEPSEEK_EMAIL", ""),
                        help="DeepSeek account email (auto-login)")
    parser.add_argument("--password", default=os.environ.get("DEEPSEEK_PASSWORD", ""),
                        help="DeepSeek account password (auto-login)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    if args.email:
        os.environ["DEEPSEEK_EMAIL"] = args.email
    if args.password:
        os.environ["DEEPSEEK_PASSWORD"] = args.password
    if args.token:
        os.environ["DEEPSEEK_TOKEN"] = args.token

    if not (os.environ.get("DEEPSEEK_EMAIL") and os.environ.get("DEEPSEEK_PASSWORD")) \
       and not os.environ.get("DEEPSEEK_TOKEN") \
       and not load_saved_token():
        print("Error: provide --email + --password, --token, or run grab_token.py first")
        exit(1)

    log.info(f"Starting DeepSeek proxy on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
