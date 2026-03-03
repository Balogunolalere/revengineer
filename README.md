<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/node.js-18+-green?logo=nodedotjs&logoColor=white" alt="Node.js">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/status-working-brightgreen" alt="Status">
</p>

# 🔧 Revengineer

> Reverse-engineered [chat.deepseek.com](https://chat.deepseek.com) — use DeepSeek R1 from your terminal, or expose it as an OpenAI-compatible API for your tools.

---

## ✨ What's Inside

| File | Description |
|------|-------------|
| [`deepseek_cli.py`](deepseek_cli.py) | Interactive terminal chat with streaming, thinking, and web search |
| [`deepseek_api.py`](deepseek_api.py) | FastAPI server — drop-in OpenAI `/v1/chat/completions` replacement |
| [`pow_solver.js`](pow_solver.js) | Fast Node.js Proof-of-Work solver (~1.5s vs ~290s in Python) |
| [`grab_token.py`](grab_token.py) | Browser-based token grabber for Google OAuth login |
| [`stress_test.py`](stress_test.py) | Rate limit & load testing tool |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USER/revengineer.git
cd revengineer

# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv if needed
uv sync                                           # creates venv + installs deps

# Or using pip
pip install -r requirements.txt
```

> **Node.js** (optional but recommended): The PoW solver runs ~100x faster with Node.  
> If `node` isn't on your PATH, the pure-Python fallback kicks in automatically.

### 2. Authenticate

<details>
<summary><b>🔑 Option A — Google Login (recommended)</b></summary>

```bash
# Opens a real browser — log in with Google, token saved automatically
python grab_token.py

# Verify it worked
python grab_token.py --validate
```

Token is saved to `~/.deepseek_token` and auto-loaded by both the CLI and API server.

</details>

<details>
<summary><b>🔑 Option B — Manual Token</b></summary>

1. Open [chat.deepseek.com](https://chat.deepseek.com) in your browser
2. Open DevTools → **Network** tab
3. Send a message, find any API request
4. Copy the `Authorization: Bearer <token>` value

```bash
# Pass directly
python deepseek_cli.py --token YOUR_TOKEN

# Or save for auto-loading
echo "YOUR_TOKEN" > ~/.deepseek_token
chmod 600 ~/.deepseek_token
```

</details>

<details>
<summary><b>🔑 Option C — Email / Password</b></summary>

> Only works if you registered directly with DeepSeek (not Google/GitHub SSO).

```bash
python deepseek_cli.py --email you@example.com --password yourpass
```

Or set in `.env`:
```env
DEEPSEEK_EMAIL=you@example.com
DEEPSEEK_PASSWORD=yourpass
```

</details>

### 3. Chat!

```bash
uv run python deepseek_cli.py
```

```
╔══════════════════════════════════════════╗
║   DeepSeek Chat CLI (reverse-engineered)  ║
╚══════════════════════════════════════════╝

You: What is the mass of the sun?
DeepSeek: The mass of the Sun is approximately 1.989 × 10³⁰ kg ...
```

---

## 💬 CLI Usage

```bash
uv run python deepseek_cli.py                  # default (thinking + search ON)
uv run python deepseek_cli.py --no-think       # disable thinking (faster responses)
uv run python deepseek_cli.py --no-search      # disable web search
uv run python deepseek_cli.py --token TOKEN    # use a specific token
```

**In-chat commands:**

| Command | Action |
|---------|--------|
| `/new` | Start a new chat session |
| `/think` | Toggle thinking mode on/off |
| `/search` | Toggle web search on/off |
| `/quit` | Exit |

### Search with Citations

When search is enabled, responses include numbered references that link to their sources:

```
DeepSeek: Python 3.14 was released in October 2025 [1] with full support
          until 2027 [3]. Key features include t-strings [4]...

📚 Sources:
  [1] Python 3.14: List Releases (VersionLog)
      https://versionlog.com/python/3.14/
  [3] Fin de support (csirt-bfc)
      https://www.csirt-bfc.fr/eol/python.html
  [4] 2025年の振り返りと2026年の展望 (Pythonエンジニア育成推進協会)
      https://www.pythonic-exam.com/archives/10810
```

---

## 🌐 API Server

Drop-in replacement for OpenAI's API. Works with **Continue**, **Cursor**, **aider**, **LangChain**, **OpenAI SDK**, etc.

### Start the server

```bash
uv run python deepseek_api.py                          # default: port 8000
uv run python deepseek_api.py --port 8080              # custom port
uv run python deepseek_api.py --host 127.0.0.1         # localhost only
```

### Use with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Use with curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

### Available Models

| Model Name | Behavior |
|-----------|----------|
| `deepseek-chat` | Standard chat |
| `deepseek-reasoner` / `deepseek-r1` | Thinking/reasoning enabled |
| `deepseek-search` | Web search enabled (citations included) |

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `GET` | `/v1/models` | List available models |
| `GET` | `/stats` | Server statistics |
| `GET` | `/health` | Health check |

---

## ⚙️ Environment Variables

All config can be set via `.env` file or exported in your shell:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPSEEK_TOKEN` | Bearer token | — |
| `DEEPSEEK_EMAIL` | Login email | — |
| `DEEPSEEK_PASSWORD` | Login password | — |
| `DEEPSEEK_TOKEN_FILE` | Path to saved token file | `~/.deepseek_token` |
| `API_HOST` | API server bind address | `0.0.0.0` |
| `API_PORT` | API server port | `8000` |

---

## 🧪 Stress Testing

```bash
# 50 requests, 3 concurrent workers
uv run python stress_test.py --total 50 --concurrent 3

# Burst test: 10 simultaneous requests
uv run python stress_test.py --burst 10

# Duration test: run for 60 seconds
uv run python stress_test.py --duration 60
```

> **Known rate limit**: ~76 requests at 3 concurrent before HTTP 202 block.

---

## 🔬 Technical Details

<details>
<summary><b>Proof of Work (DeepSeekHashV1)</b></summary>

DeepSeek requires a PoW challenge before each completion request. The algorithm is a **custom Keccak-256** variant:

- **23 rounds** (rounds 1–23, skipping round 0) — standard Keccak uses 24 rounds (0–23)
- Padding byte `0x06` (same as SHA-3)
- Brute-force: find integer `i` where `keccak(salt + "_" + expire_at + "_" + i) == challenge`
- Typical difficulty: ~144,000

The Node.js solver (`pow_solver.js`) handles this in ~1.5s. The Python fallback works but takes ~290s.

</details>

<details>
<summary><b>SSE Stream Format</b></summary>

DeepSeek uses a **JSON-patch** based SSE format, NOT OpenAI-style:

```json
{"p": "response/fragments", "o": "APPEND", "v": [{"type": "THINK", "content": "Let me..."}]}
{"p": "response/fragments/-1/content", "v": " think about this"}
{"v": " more text"}
```

Fields:
- `p` — JSON path (e.g., `response/fragments/-1/content`)
- `o` — operation (`APPEND`, `SET`, etc.)
- `v` — value

Fragment types: `SEARCH`, `THINK`, `RESPONSE`

</details>

<details>
<summary><b>Search & Citations</b></summary>

When search is enabled, the SSE stream includes a `SEARCH` fragment with queries and results:

```json
{
  "p": "response/fragments/-1/results",
  "v": [
    {
      "url": "https://example.com/article",
      "title": "Article Title",
      "cite_index": 1,
      "site_name": "Example"
    }
  ]
}
```

The response text contains `[citation:N]` tags that map to the `cite_index` values. These are converted to clean `[N]` references with a sources list appended.

</details>

---

## 📁 Project Structure

```
revengineer/
├── deepseek_cli.py      # Interactive terminal client
├── deepseek_api.py      # FastAPI OpenAI-compatible proxy
├── pow_solver.js         # Fast Node.js PoW solver
├── grab_token.py         # Playwright browser token grabber
├── stress_test.py        # Load testing tool
├── cookbook/
│   ├── swarm/            # 🐝 Parallel agent orchestration system
│   │   ├── models.py     # Data models with DAG validation
│   │   ├── config.py     # SwarmConfig with SWARM_* env vars
│   │   ├── engine.py     # Concurrent execution engine
│   │   ├── orchestrator.py  # LLM decomposition + synthesis
│   │   ├── renderer.py   # Live spinner + terminal output
│   │   └── cli.py        # CLI entry point
│   ├── examples/         # Ready-to-run example scripts
│   └── tests/            # 64 unit tests
├── pyproject.toml        # uv project config
├── requirements.txt      # pip dependencies
├── .env                  # Your secrets (git-ignored)
└── .gitignore
```

---

## ⚠️ Disclaimer

This project reverse-engineers a public web interface for personal/educational use. It is **not** affiliated with DeepSeek. Use responsibly and respect their terms of service.
