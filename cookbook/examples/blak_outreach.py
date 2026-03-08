"""BlakStudio.dev — Intelligent Instagram DM Outreach Bot

Two discovery modes that work together:
  A) TIMELINE — scans your home feed (paginates to pull more posts)
  B) HASHTAG  — searches hashtags you specify, finds businesses there

For every discovered account it:
1. Fetches their full business profile
2. Uses AI to qualify them (are they a business that needs web/email/branding?)
3. Generates a hyper-personalized DM pitching the exact BlakStudio service they need
4. Sends the DM (dedup tracked — never DMs the same person twice)
5. Likes the post + leaves a smart comment (vision-powered)
6. Optionally follows qualified accounts

Usage:
    export IG_USERNAME="you@example.com"
    export IG_PASSWORD="your-password"
    export OPENROUTER_API_KEY="sk-or-v1-..."     # primary key
    export OPENROUTER_API_KEY_2="sk-or-v1-..."   # optional 2nd account
    export OPENROUTER_API_KEY_3="sk-or-v1-..."   # optional 3rd account
    export NVIDIA_API_KEY="nvapi-..."             # NVIDIA NIM free preview

    # Run once (scan feed, qualify, DM, exit)
    python -m cookbook.examples.blak_outreach --once

    # Run continuously (every 10 min by default)
    python -m cookbook.examples.blak_outreach

    # Custom interval + no commenting (DM only)
    python -m cookbook.examples.blak_outreach --interval 900 --no-comment

    # DM dry-run (qualify + generate DM but don't send)
    python -m cookbook.examples.blak_outreach --dry-run

    # Hashtag discovery mode — search specific hashtags for leads
    python -m cookbook.examples.blak_outreach --hashtags "smallbusiness,lagosvendor,africanfashion"

    # Disable auto-follow
    python -m cookbook.examples.blak_outreach --no-follow

    # Disable auto-hashtag discovery (timeline feed only)
    python -m cookbook.examples.blak_outreach --no-hashtags

    # Full combo: manual hashtags + no follow, dry run
    python -m cookbook.examples.blak_outreach --hashtags "webdesign,startup" --no-follow --dry-run
"""

import argparse
import asyncio
import collections
import datetime
import ipaddress
import json
import logging
import math
import os
import random
import re
import socket
import sys
import time
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()  # auto-load .env from project root

import httpx

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.rule import Rule
from rich.align import Align
from rich import box

from cookbook.swarm.instagrapi_bridge import InstagrapiBridge
from cookbook.swarm.config import SwarmConfig

console = Console()
log = logging.getLogger("blak_outreach")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

# Suppress noisy instagrapi graphql 401 retry logs
logging.getLogger("instagrapi").setLevel(logging.WARNING)
logging.getLogger("public_request").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


# ── Shared constants (Fix 13: single definition) ─────────────────

SOCIAL_DOMAINS = [
    "instagram.com", "facebook.com", "twitter.com", "tiktok.com",
    "youtube.com", "wa.me", "t.me", "snapchat.com", "whatsapp.com",
    "threads.net", "x.com", "linkedin.com", "pinterest.com",
    "play.google.com", "apps.apple.com", "open.spotify.com",
    "music.apple.com", "soundcloud.com", "tidal.com",
]

LINK_IN_BIO_SERVICES = [
    "linktr.ee", "linktree.com", "bio.link", "linkbio.co", "tap.bio",
    "campsite.bio", "solo.to", "carrd.co", "beacons.ai", "lnk.bio",
    "milkshake.app", "stan.store", "hoo.be", "snipfeed.co", "flowpage.com",
]

# Maximum post age — ignore posts older than this
MAX_POST_AGE_DAYS = 180  # 6 months


# ── BlakStudio services ──────────────────────────────────────────

BLAK_SERVICES = """
BlakStudio.dev offers:
1. **Custom Websites** — Professional business websites, portfolios, landing pages. From ₦150k / $150.
2. **Business Email Setup** — Custom domain email (e.g., info@yourbrand.com) with Google Workspace or Zoho. From ₦30k / $30.
3. **Brand Identity** — Logo design, brand guidelines, social media templates. From ₦50k / $50.
4. **E-Commerce Stores** — Shopify, WooCommerce, or custom stores with payment integration. From ₦200k / $200.
5. **SEO & Digital Marketing** — Google rankings, social media management, paid ads. From ₦80k / $80.
6. **Web/Mobile App Development** — Custom apps, dashboards, booking systems. From ₦300k / $300.

Website: blakstudio.dev
"""

# ── AI Prompts ────────────────────────────────────────────────────

QUALIFY_SYSTEM = f"""You are a sales qualification assistant for BlakStudio.dev, a web development and digital services agency.

{BLAK_SERVICES}

Given an Instagram business profile (with follower count, bio, category, website status, email status, etc.), determine:
1. Are they a legitimate business/brand/creator that could benefit from BlakStudio's services?
2. What specific service(s) do they need most?
3. Should we reach out? (YES or NO)

Criteria for YES:
- They ARE a business, creator, or brand (not a regular personal account)
- They are missing a website, OR have a basic linktree/link-in-bio instead of a proper site
- OR they don't have a business email (using gmail/yahoo/outlook instead of @theirbrand.com)
- OR they seem to need better branding/digital presence based on their bio
- They have at least 100+ followers (worth reaching out to)
- Their account is public

Criteria for NO:
- Regular personal accounts (not a business)
- Already have a professional website
- Already have custom business email
- Private accounts, spam, or fake accounts
- Under 100 followers (too small)
- Major brands/verified accounts that obviously have agencies already

Respond in this exact JSON format (no markdown, no code fences):
{{"qualify": "YES" or "NO", "reason": "1-2 sentence explanation", "services_needed": ["website", "email", "branding", "ecommerce", "seo", "app"], "priority": "high" or "medium" or "low"}}"""


DM_SYSTEM = f"""You are a friendly, professional outreach specialist for BlakStudio.dev.

{BLAK_SERVICES}

Write a short, personalized Instagram DM (3-5 sentences max). The message should:
- Open with something specific about THEIR business/brand (reference their bio, category, or what they do)
- Identify the specific gap (no website, weak online presence, no business email, etc.)
- Pitch the 1-2 most relevant BlakStudio services
- Include a soft CTA (check out blakstudio.dev, or "want me to send a quick proposal?")
- Sound natural and conversational, NOT like a template
- NO emojis spam (0-2 max)
- Under 500 characters (Instagram DM sweet spot)
- IMPORTANT: If the qualification data includes "has_website", do NOT pitch a website. They already have one. Focus ONLY on the services listed in "services_needed".
- NEVER suggest they need a website if they already have one.

If you cannot write a genuine personalized message, respond with SKIP.

Respond with ONLY the DM text, nothing else."""

HASHTAG_SYSTEM = f"""You are a growth strategist for BlakStudio.dev.

{BLAK_SERVICES}

Your job is to find Instagram hashtags where potential BlakStudio customers hang out.
Think about: small businesses, vendors, freelancers, startups, creators — anyone who
might need a website, business email, branding, e-commerce store, or digital marketing.

Focus on:
- Nigerian / African small business hashtags (Lagos, Abuja, Accra, Nairobi, etc.)
- Industry-specific tags (real estate, fashion, food, beauty, interior design, photography, events)
- Entrepreneur / hustler / side-business tags
- "Vendor" and "shop" tags
- Avoid mega-hashtags with 100M+ posts (too noisy) — sweet spot is 10K-5M posts

Return EXACTLY 8 hashtags as a JSON array of strings (no #, no markdown, no explanation).
Example: ["lagosvendor", "abujabusiness", "nigerianfashiondesigner", ...]

Vary your picks each time — mix popular ones with niche ones."""

COMMENT_SYSTEM = """You are a friendly Instagram user commenting on a post. Write a SHORT, genuine comment (1-2 sentences max, under 150 chars).

Guidelines:
- ALWAYS write a comment. You should almost never skip.
- Reference something specific from the caption or image description.
- 0-2 emojis max. No emoji spam.
- Match the vibe: professional for business, casual for personal.
- Never be promotional or salesy.
- If the post shows products (shoes, clothes, food, etc.), compliment the aesthetics, colors, or design. NEVER ask about price, size, or availability.
- Only respond with the single word SKIP if the post has absolutely zero meaningful content (completely empty, broken, or unintelligible).

Respond with ONLY the comment text. Do NOT add quotes around it."""

SELECT_SYSTEM = (
    "You are an Instagram engagement strategist for BlakStudio.dev, "
    "a web development and digital services agency.\n\n"
    "Given a list of Instagram posts, analyze each and decide which are worth engaging with.\n\n"
    "Prioritize:\n"
    "- Small businesses, vendors, creators who might need web/digital services\n"
    "- Posts with genuine, substantive content that allows a meaningful comment\n"
    "- Posts from businesses or professionals (not personal drama, memes, spam)\n"
    "- Active accounts with real engagement potential\n\n"
    "For each post, assign ONE action: ENGAGE, LIKE, or SKIP.\n\n"
    "Respond with ONLY a valid JSON array and absolutely NO other text or markdown formatting.\n"
    "Example format:\n"
    '[{"i": 0, "a": "ENGAGE"}, {"i": 1, "a": "LIKE"}, {"i": 2, "a": "SKIP"}]'
)


# ── LLM helpers ───────────────────────────────────────────────────

# ── Vision API config (OpenRouter free + NVIDIA NIM free preview) ─
# Two providers rotate together for maximum throughput & reliability.
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS: list[str] = [
    "google/gemma-3-4b-it:free",             # 4B, 32K ctx — confirmed working
    "mistralai/mistral-small-3.1-24b-instruct:free",  # 24B, 128K ctx
    "google/gemma-3-12b-it:free",            # 12B, 32K ctx
]

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_VISION_MODELS: list[str] = [
    "meta/llama-4-maverick-17b-128e-instruct",  # 400B MoE — best vision
    "google/gemma-3-27b-it",                     # Gemma 3 27B vision
    "meta/llama-4-scout-17b-16e-instruct",       # Llama 4 Scout vision
]

# ── Multi-key rotation ────────────────────────────────────────────
# OpenRouter: OPENROUTER_API_KEY, OPENROUTER_API_KEY_2 … _19
# NVIDIA:     NVIDIA_API_KEY (single key, free preview ~1000 req/day)
def _load_openrouter_keys() -> list[str]:
    keys = []
    primary = os.environ.get("VISION_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    if primary:
        keys.append(primary)
    for i in range(2, 20):
        k = os.environ.get(f"OPENROUTER_API_KEY_{i}", "")
        if k:
            keys.append(k)
    return keys

def _load_nvidia_keys() -> list[str]:
    k = os.environ.get("NVIDIA_API_KEY", "")
    return [k] if k else []

OPENROUTER_API_KEYS: list[str] = _load_openrouter_keys()
NVIDIA_API_KEYS: list[str] = _load_nvidia_keys()

# Build a flat roster of (base_url, model, api_key, extra_headers) slots
VisionSlot = tuple[str, str, str, dict[str, str]]

def _build_vision_slots() -> list[VisionSlot]:
    slots: list[VisionSlot] = []
    or_headers = {"HTTP-Referer": "https://blakstudio.dev", "X-Title": "BlakStudio Instagram Bot"}
    for model in OPENROUTER_MODELS:
        for key in OPENROUTER_API_KEYS:
            slots.append((OPENROUTER_API_BASE, model, key, or_headers))
    for model in NVIDIA_VISION_MODELS:
        for key in NVIDIA_API_KEYS:
            slots.append((NVIDIA_API_BASE, model, key, {}))
    return slots

VISION_SLOTS: list[VisionSlot] = _build_vision_slots()
_slot_idx = 0  # round-robin index across all slots
_slot_lock = asyncio.Lock()  # Fix 5: protect round-robin from concurrent access

# backward compat
VISION_API_KEYS: list[str] = OPENROUTER_API_KEYS or [""]
VISION_API_KEY = VISION_API_KEYS[0]

VISION_DESCRIBE_PROMPT = """Describe this Instagram post image in 1-2 sentences. Focus on:
- What's shown (people, objects, scenery, food, fashion, etc.)
- The mood/vibe (professional, casual, celebratory, etc.)
- Any text overlays visible
Be concise and factual."""


# ── Shared HTTP client (Fix 4) ───────────────────────────────────

_http_pool: httpx.AsyncClient | None = None

async def _get_http_client() -> httpx.AsyncClient:
    """Return a shared httpx client with connection pooling."""
    global _http_pool
    if _http_pool is None or _http_pool.is_closed:
        _http_pool = httpx.AsyncClient(timeout=30, follow_redirects=True)
    return _http_pool


# ── SSRF protection (Fix 3) ──────────────────────────────────────

def _is_safe_url(url: str) -> bool:
    """Return False if URL targets private/reserved/internal IPs."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        infos = socket.getaddrinfo(hostname, None)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return False
        return True
    except Exception:
        return False


# ── Instagram Rate Intelligence ──────────────────────────────────
#
# Instead of reacting to 403/429 errors, this system *understands*
# Instagram's rate limits and proactively stays under them — like a
# human who knows not to like 100 posts in 5 minutes.
#
# Key ideas:
#   1. Sliding-window action log — knows exactly how many actions
#      were performed in the last hour / last day.
#   2. Budget model — encodes Instagram's known limits and warns/
#      pauses BEFORE hitting them.
#   3. Human-like timing — Gaussian delays, micro-breaks, session
#      fatigue. A bot that acts at uniform 2s intervals is obvious.
#   4. Session consciousness — prints its own awareness of how
#      close it is to each limit.
# ─────────────────────────────────────────────────────────────────


class InstagramRateLimiter:
    """Proactive rate-limit engine that models Instagram's enforcement.

    Instagram enforces approximate limits (varies by account age/trust):
    ┌────────────┬──────────┬──────────┬──────────────────────────┐
    │ Action     │ /hour    │ /day     │ Notes                    │
    ├────────────┼──────────┼──────────┼──────────────────────────┤
    │ Like       │ ~60      │ ~300     │ Strictest on new accts   │
    │ Comment    │ ~20      │ ~100     │ Duplicate text triggers   │
    │ DM         │ ~15      │ ~50      │ Hardest — 403 at ~10-20  │
    │ Follow     │ ~20      │ ~100     │ Follow/unfollow flagged  │
    │ Profile    │ ~80      │ ~500     │ Scraping detected        │
    │ Search     │ ~30      │ ~200     │ Hashtag search           │
    └────────────┴──────────┴──────────┴──────────────────────────┘

    The bot stays under 75% of these limits and starts throttling
    aggressively at 85%.  At 95% it pauses proactively.
    """

    # Known Instagram rate ceilings (conservative estimates)
    LIMITS = {
        #               (per_hour, per_day)
        "like":         (50,  250),
        "comment":      (18,   80),
        "dm":           (10,   40),
        "follow":       (15,   80),
        "profile_view": (60,  400),
        "search":       (25,  150),
    }

    # Thresholds (fraction of limit)
    WARN_AT   = 0.70   # start slowing down
    SLOW_AT   = 0.85   # add significant delays
    PAUSE_AT  = 0.95   # stop doing this action entirely

    # Base delays between actions (seconds) — the "relaxed human" pace
    BASE_DELAY = {
        "like":         (3.0,   8.0),    # humans scroll, pause, like
        "comment":      (12.0,  25.0),   # writing takes time
        "dm":           (30.0,  60.0),   # composing a message is slow
        "follow":       (10.0,  20.0),   # deliberate action
        "profile_view": (2.0,   5.0),    # quick glances
        "search":       (5.0,   12.0),   # browsing
    }

    # Micro-break: every N actions, take a longer pause (like putting phone down)
    BREAK_EVERY = 12           # actions before a micro-break
    BREAK_DURATION = (30, 90)  # seconds — humans check other apps, stretch

    # Session fatigue: the longer the session, the slower humans get
    FATIGUE_ONSET_MIN = 20     # fatigue kicks in after 20 min
    FATIGUE_MAX_MULT  = 2.0    # at most 2x slower after hours of use

    def __init__(self):
        # Sliding window: action -> deque of timestamps
        self._log: dict[str, collections.deque] = {
            action: collections.deque() for action in self.LIMITS
        }
        self._session_start = time.time()
        self._total_actions = 0       # all actions combined
        self._consecutive_fails: dict[str, int] = {a: 0 for a in self.LIMITS}
        self._forced_pause: dict[str, float] = {a: 0.0 for a in self.LIMITS}  # pause until timestamp

        # ── Dynamic session planning ──
        self._session_duration: float | None = None   # seconds, None = unlimited
        self._session_end: float | None = None
        self._session_budget: dict[str, int] = {}     # max actions for this session
        self._session_delays: dict[str, tuple[float, float]] = {}  # paced (low, high) per action

    # ── Session Duration Planner ─────────────────────────────────

    @staticmethod
    def parse_duration(s: str) -> float:
        """Parse human duration string → seconds.

        Examples: '5m' → 300, '1h' → 3600, '2h30m' → 9000, '90s' → 90
        """
        import re as _re
        s = s.strip().lower()
        if not s:
            return 0.0
        total = 0.0
        for val, unit in _re.findall(r'(\d+(?:\.\d+)?)\s*([hms])', s):
            v = float(val)
            if unit == 'h':
                total += v * 3600
            elif unit == 'm':
                total += v * 60
            elif unit == 's':
                total += v
        if total == 0.0:
            # Bare number → assume minutes
            try:
                total = float(s) * 60
            except ValueError:
                pass
        return total

    def plan_session(self, duration_sec: float):
        """Plan the entire session: compute per-action budgets & pacing.

        Given a session duration, figures out:
        1. How many of each action fit safely (pro-rated from hourly limits)
        2. What delay between each action keeps them evenly distributed
        3. When the session must end

        This is the 'intelligence' — instead of static delays, the bot
        reasons: "I have 5 minutes and can do 4 DMs max this hour, so
        I should space them ~75s apart and stop after 4."
        """
        self._session_duration = duration_sec
        self._session_end = self._session_start + duration_sec
        dur_hours = duration_sec / 3600

        for action, (h_limit, d_limit) in self.LIMITS.items():
            # Pro-rate from hourly limit (the burst-rate constraint).
            # For short sessions (< 1h) the hourly limit drives the budget.
            # We cap at 70% of the daily limit for multi-hour sessions so
            # a single run never blows the full-day allowance.
            from_hourly = h_limit * dur_hours * self.WARN_AT
            daily_cap   = int(d_limit * self.WARN_AT)
            budget = min(max(1, int(from_hourly)), daily_cap)
            self._session_budget[action] = budget

            # Pacing: spread actions evenly across the session
            if budget >= 2:
                ideal_gap = duration_sec / budget
                # Add 20% jitter range around the ideal gap
                low = ideal_gap * 0.80
                high = ideal_gap * 1.20
            else:
                # Only 1 action — do it sometime in the session
                low, high = self.BASE_DELAY.get(action, (3.0, 8.0))
            self._session_delays[action] = (low, high)

        log.info(
            f"Session planned: {duration_sec / 60:.0f} min, "
            f"budgets: {self._session_budget}"
        )

    def session_plan_table(self) -> dict[str, dict]:
        """Return the session plan for display."""
        plan = {}
        for action in self.LIMITS:
            budget = self._session_budget.get(action, "∞")
            used = self.count_in_window(action, self._session_duration or 3600)
            delays = self._session_delays.get(action)
            avg_gap = (delays[0] + delays[1]) / 2 if delays else 0
            plan[action] = {
                "budget": budget,
                "used": used,
                "remaining": (budget - used) if isinstance(budget, int) else "∞",
                "avg_gap_sec": avg_gap,
            }
        return plan

    def adjust_session_clock(self, seconds: float):
        """Add time to the session clock (e.g. to exclude scraping time from limits).

        Called after scraping/selection/preparation phases so that
        the duration those phases took doesn't count against execution time.
        """
        if self._session_duration is None:
            return
        self._session_start += seconds
        self._session_end += seconds
        log.info(
            f"Session clock shifted +{seconds:.1f}s to account for prep time"
        )

    def time_remaining(self) -> float | None:
        """Seconds left in session, or None if unlimited."""
        if self._session_end is None:
            return None
        return max(0.0, self._session_end - time.time())

    def session_expired(self) -> bool:
        """True if a duration was set and time is up."""
        if self._session_end is None:
            return False
        return time.time() >= self._session_end

    def all_budgets_exhausted(self, actions: list[str]) -> bool:
        """Return True if EVERY requested action has exhausted its session budget."""
        if not self._session_duration:
            return False  # No budget in unlimited sessions
        for action in actions:
            allowed, _ = self.should_act(action)
            if allowed:
                return False
        return True

    # ── Core: record & query ────────────────────────────────────

    def record(self, action: str):
        """Record that an action just happened."""
        now = time.time()
        self._log[action].append(now)
        self._total_actions += 1
        self._consecutive_fails[action] = 0
        # Prune old entries (older than 25 hours — keep some buffer)
        cutoff = now - 90000
        for dq in self._log.values():
            while dq and dq[0] < cutoff:
                dq.popleft()

    def record_fail(self, action: str):
        """Record a failed attempt (403/429). Triggers reactive pause."""
        self._consecutive_fails[action] = self._consecutive_fails.get(action, 0) + 1
        fails = self._consecutive_fails[action]
        if fails >= 2:
            # Exponential backoff: 10min, 20min, 40min, capped at 60min
            cooldown = min(3600, 600 * (2 ** (fails - 2)))
            self._forced_pause[action] = time.time() + cooldown
            log.warning(
                f"🛑 {action.upper()} rate-limited — forced pause for "
                f"{cooldown // 60:.0f} min (consecutive fails: {fails})"
            )

    def count_in_window(self, action: str, window_sec: float) -> int:
        """Count how many times this action was performed in last `window_sec`."""
        cutoff = time.time() - window_sec
        return sum(1 for ts in self._log.get(action, []) if ts > cutoff)

    def hourly(self, action: str) -> int:
        """Actions in the last 60 minutes."""
        return self.count_in_window(action, 3600)

    def daily(self, action: str) -> int:
        """Actions in the last 24 hours."""
        return self.count_in_window(action, 86400)

    # ── Decision: should we do this action? ─────────────────────

    def utilization(self, action: str) -> tuple[float, float]:
        """Return (hourly_fraction, daily_fraction) of limit used. 0.0–1.0+"""
        h_limit, d_limit = self.LIMITS.get(action, (50, 200))
        h_used = self.hourly(action) / h_limit
        d_used = self.daily(action) / d_limit
        return (h_used, d_used)

    def should_act(self, action: str) -> tuple[bool, str]:
        """Check if we should perform this action right now.

        Returns (allowed: bool, reason: str).
        This is the brain — it proactively prevents rate limits.
        """
        # 0) Check session time budget
        if self.session_expired():
            return False, "session duration reached — winding down"

        # 0b) Check session action budget
        if action in self._session_budget:
            used = self.count_in_window(action, self._session_duration or 3600)
            if used >= self._session_budget[action]:
                return False, f"session budget exhausted ({used}/{self._session_budget[action]})"

        # 1) Check reactive forced pauses (from actual 403/429 hits)
        if time.time() < self._forced_pause.get(action, 0):
            remaining = self._forced_pause[action] - time.time()
            return False, f"forced pause ({remaining / 60:.0f} min left)"

        # 2) Check proactive limits
        h_frac, d_frac = self.utilization(action)
        worst = max(h_frac, d_frac)

        if worst >= self.PAUSE_AT:
            period = "hourly" if h_frac >= d_frac else "daily"
            pct = worst * 100
            return False, f"at {pct:.0f}% of {period} limit — pausing to avoid ban"

        # Allowed — but we might slow down
        return True, ""

    def budget_remaining(self, action: str) -> dict:
        """Return how many more actions are safe this hour/day."""
        h_limit, d_limit = self.LIMITS.get(action, (50, 200))
        h_used = self.hourly(action)
        d_used = self.daily(action)
        # Safe = up to WARN_AT threshold
        h_safe = max(0, int(h_limit * self.WARN_AT) - h_used)
        d_safe = max(0, int(d_limit * self.WARN_AT) - d_used)
        return {
            "hourly_used": h_used, "hourly_limit": h_limit, "hourly_safe": h_safe,
            "daily_used": d_used, "daily_limit": d_limit, "daily_safe": d_safe,
        }

    # ── Timing: how long to wait ────────────────────────────────

    def get_delay(self, action: str) -> float:
        """Calculate a human-like delay before the next action.

        If a session duration is set, uses paced delays computed from
        the session plan. Otherwise falls back to base delays with
        utilization scaling.

        Incorporates:
        - Session-aware pacing (dynamic, not static)
        - Utilization scaling (slower when approaching limits)
        - Session fatigue (slower the longer we've been running)
        - Gaussian jitter (humans aren't uniform)
        - Micro-breaks every N actions (put the phone down)
        """
        # ── Session-paced delays (dynamic) ──
        if action in self._session_delays:
            low, high = self._session_delays[action]
        else:
            low, high = self.BASE_DELAY.get(action, (3.0, 8.0))

        # If session has a time budget, adapt delays to remaining time
        remaining = self.time_remaining()
        if remaining is not None and action in self._session_budget:
            budget = self._session_budget[action]
            used = self.count_in_window(action, self._session_duration or 3600)
            actions_left = max(1, budget - used)
            # Re-pace: how long should each remaining action take?
            ideal_gap = remaining / actions_left
            low = ideal_gap * 0.75
            high = ideal_gap * 1.25

        # ── Utilization scaling ──
        h_frac, d_frac = self.utilization(action)
        worst = max(h_frac, d_frac)
        if worst >= self.SLOW_AT:
            # Approaching limit — dramatically slow down
            # At 85% → 2x, at 90% → 3x, at 94% → 4.5x
            scale = 1.0 + ((worst - self.SLOW_AT) / (self.PAUSE_AT - self.SLOW_AT)) * 4.0
            low *= scale
            high *= scale
        elif worst >= self.WARN_AT:
            # Getting warm — gently slow down
            scale = 1.0 + (worst - self.WARN_AT) * 2.0
            low *= scale
            high *= scale

        # ── Session fatigue ──
        session_min = (time.time() - self._session_start) / 60
        if session_min > self.FATIGUE_ONSET_MIN:
            # Logarithmic fatigue — gets slower but caps out
            fatigue = min(
                self.FATIGUE_MAX_MULT,
                1.0 + math.log1p((session_min - self.FATIGUE_ONSET_MIN) / 30) * 0.3
            )
            low *= fatigue
            high *= fatigue

        # ── Gaussian jitter (humans cluster around a mean, not uniform) ──
        mean = (low + high) / 2
        std = (high - low) / 4  # 95% of values within [low, high]
        delay = max(low * 0.8, random.gauss(mean, std))  # floor at 80% of low

        # ── Micro-break ──
        if self._total_actions > 0 and self._total_actions % self.BREAK_EVERY == 0:
            break_dur = random.uniform(*self.BREAK_DURATION)
            log.info(
                f"☕ Micro-break: pausing {break_dur:.0f}s after "
                f"{self._total_actions} actions (human behavior)"
            )
            delay += break_dur

        return delay

    def forced_pause_remaining(self, action: str) -> float:
        return max(0.0, self._forced_pause.get(action, 0) - time.time())

    # ── Reporting ───────────────────────────────────────────────

    def status_summary(self) -> dict[str, dict]:
        """Return a snapshot of all action budgets for display."""
        summary = {}
        for action in self.LIMITS:
            b = self.budget_remaining(action)
            h_frac, d_frac = self.utilization(action)
            forced_remain = self.forced_pause_remaining(action)
            # Session budget info
            sess_budget = self._session_budget.get(action)
            sess_used = self.count_in_window(action, self._session_duration or 3600) if sess_budget else None
            summary[action] = {
                **b,
                "hourly_pct": h_frac * 100,
                "daily_pct": d_frac * 100,
                "forced_pause_min": forced_remain / 60 if forced_remain > 0 else 0,
                "session_budget": sess_budget,
                "session_used": sess_used,
            }
        return summary

    def session_duration_min(self) -> float:
        return (time.time() - self._session_start) / 60


_ig = InstagramRateLimiter()


# ── Legacy throttle (still used for vision/LLM error backoff) ────

class _ThrottleState:
    """Tracks consecutive errors and adjusts sleep durations adaptively."""
    def __init__(self):
        self.consecutive_errors = 0
        self.multiplier = 1.0

    def on_success(self):
        self.consecutive_errors = 0
        self.multiplier = max(1.0, self.multiplier * 0.8)

    def on_error(self):
        self.consecutive_errors += 1
        self.multiplier = min(5.0, self.multiplier * 1.5)

    def sleep(self, low: float, high: float) -> float:
        return random.uniform(low * self.multiplier, high * self.multiplier)

_throttle = _ThrottleState()


# ── DM retry queue ────────────────────────────────────────────────

_DM_RETRY_FILE = "dm_retry_queue.json"


def _load_dm_retry_queue() -> list[dict]:
    """Load the retry queue from disk."""
    try:
        with open(_DM_RETRY_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_dm_retry_queue(queue: list[dict]):
    with open(_DM_RETRY_FILE, "w") as f:
        json.dump(queue, f, indent=2)


def _enqueue_dm_retry(user_id: str, username: str, dm_text: str):
    """Add a failed DM to the retry queue (dedup by user_id)."""
    queue = _load_dm_retry_queue()
    existing_ids = {item["user_id"] for item in queue}
    if user_id in existing_ids:
        return
    queue.append({
        "user_id": user_id,
        "username": username,
        "dm_text": dm_text,
        "failed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "retries": 0,
    })
    _save_dm_retry_queue(queue)


def _remove_from_retry_queue(user_id: str):
    queue = _load_dm_retry_queue()
    queue = [item for item in queue if item["user_id"] != user_id]
    _save_dm_retry_queue(queue)


# ── JSON extraction (Fix 8) ──────────────────────────────────────

def _extract_json(raw: str, expect_array: bool = False):
    """Extract JSON object/array from LLM response, ignoring markdown fences/prose."""
    if not raw:
        return None
    pattern = r'\[.*\]' if expect_array else r'\{.*\}'
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def generate_fallback(candidates: list[dict], like_budget: int, comment_budget: int) -> list[dict]:
    """Fallback when LLM selection fails: assign ENGAGE to top items within both budgets."""
    selected = []
    engage_count = 0
    like_count = 0
    for c in candidates:
        if engage_count < comment_budget and (engage_count + like_count) < like_budget:
            selected.append(dict(c, action="ENGAGE"))
            engage_count += 1
        elif (engage_count + like_count) < like_budget:
            selected.append(dict(c, action="LIKE"))
            like_count += 1
        else:
            break
    return selected


# ── Stats / dedup helpers (Fixes 2, 12, 15) ──────────────────────


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed time for display."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def _new_stats() -> dict:
    return {
        "posts": 0, "liked": 0, "skipped": 0, "qualified": 0,
        "dm_sent": 0, "dm_drafted": 0, "dm_skipped": 0, "dm_retried": 0,
        "commented": 0, "followed": 0, "failed": 0,
        "vision_failures": 0, "rate_limited": 0,
    }

async def _load_dedup_sets(bridge: InstagrapiBridge) -> dict:
    """Load liked/commented/DMed sets once per pass (Fix 2)."""
    liked_data = json.loads(await bridge.get_liked_posts())
    commented_data = json.loads(await bridge.get_commented_posts())
    dm_data = json.loads(await bridge.get_dm_sent())
    return {
        "liked": set(str(i) for i in liked_data.get("liked_ids", [])),
        "commented": set(str(i) for i in commented_data.get("commented_ids", [])),
        "dmed": set(str(i) for i in dm_data.get("dm_sent_ids", [])),
    }


# ── LLM helpers ───────────────────────────────────────────────────

async def llm_call(config: SwarmConfig, system: str, user_msg: str, max_tokens: int = 200, temp: float = 0.7) -> str | None:
    """Generic DeepSeek text completion call (Fix 4: uses shared HTTP client)."""
    api_base = config.api_base or "http://localhost:8000/v1"
    try:
        client = await _get_http_client()
        resp = await client.post(
            f"{api_base}/chat/completions",
            json={
                "model": config.default_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": temp,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return None


async def describe_image(image_url: str, stats: dict | None = None) -> str | None:
    """Use OpenRouter + NVIDIA NIM vision models to describe an image.

    Rotates through all provider/model/key slots round-robin (Fix 5: lock-protected).
    On 429 or failure, advances to the next slot.
    Fix 15: tracks vision_failures in stats when all slots are exhausted.
    """
    global _slot_idx
    if not VISION_SLOTS or not image_url:
        return None

    # Fix 3: SSRF check on image URL
    if not _is_safe_url(image_url):
        log.warning(f"Blocked unsafe image URL: {image_url[:80]}")
        return None

    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_DESCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        "max_tokens": 150,
        "stream": False,
    }

    client = await _get_http_client()

    for attempt in range(len(VISION_SLOTS)):
        # Fix 5: lock protects round-robin index from concurrent access
        async with _slot_lock:
            base_url, model, api_key, extra_headers = VISION_SLOTS[_slot_idx % len(VISION_SLOTS)]
            _slot_idx = (_slot_idx + 1) % len(VISION_SLOTS)

        headers = {"Authorization": f"Bearer {api_key}", **extra_headers}
        provider = "NVIDIA" if "nvidia" in base_url else "OR"
        short_name = model.split('/')[-1]
        try:
            resp = await client.post(
                f"{base_url}/chat/completions",
                json={"model": model, **payload},
                headers=headers,
                timeout=45,
            )
            if resp.status_code == 429:
                log.warning(f"Vision 429 on {short_name} ({provider}), rotating...")
                await asyncio.sleep(1)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"].get("content")
            if not content:
                continue
            desc = content.strip()
            console.print(f"         [dim]👁️  Vision ({provider}/{short_name}): {desc[:70]}...[/dim]")
            return desc
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(1)
                continue
            log.warning(f"Vision failed ({provider}/{short_name}): {e}")
            continue
        except Exception as e:
            log.warning(f"Vision failed ({provider}/{short_name}): {e}")
            continue

    # Fix 15: log + track when all vision slots are exhausted
    log.warning("All vision slots exhausted — no image description available")
    if stats is not None:
        stats["vision_failures"] = stats.get("vision_failures", 0) + 1
    return None


async def verify_no_website(username: str, full_name: str, biography: str, external_url: str) -> dict:
    """Verify a business truly doesn't have a website by checking their links more carefully.

    Returns: {"has_website": bool, "url_found": str, "method": str}
    Fix 3: SSRF-safe — validates URLs against private IPs before fetching.
    Fix 4: uses shared HTTP client.
    Fix 13: uses module-level SOCIAL_DOMAINS / LINK_IN_BIO_SERVICES.
    """
    client = await _get_http_client()

    # 1) If they have a real external_url, check if it's a legit domain (not linktree etc.)
    if external_url:
        url_lower = external_url.lower()
        is_link_in_bio = any(svc in url_lower for svc in LINK_IN_BIO_SERVICES)
        if not is_link_in_bio:
            # They have a real website
            return {"has_website": True, "url_found": external_url, "method": "instagram_profile"}

        # It's a link-in-bio page — scrape it for real website links (with SSRF check)
        if _is_safe_url(external_url):
            try:
                resp = await client.get(external_url, timeout=12)
                if resp.status_code < 400:
                    page = resp.text.lower()
                    href_pattern = r'href=["\']?(https?://[^"\'<>\s]+)'
                    found_urls = re.findall(href_pattern, page)
                    for href in found_urls:
                        href_lower = href.lower()
                        if any(s in href_lower for s in SOCIAL_DOMAINS + LINK_IN_BIO_SERVICES):
                            continue
                        if any(href_lower.endswith(ext) for ext in [".png", ".jpg", ".svg", ".css", ".js"]):
                            continue
                        return {"has_website": True, "url_found": href, "method": "linkinbio_scrape"}
            except Exception:
                pass  # If scrape fails, continue with other checks

    # 2) Check their bio for website mentions
    if biography:
        bio_lower = biography.lower()
        domain_pattern = r'(?:https?://)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?)'
        domains_in_bio = re.findall(domain_pattern, bio_lower)
        for domain in domains_in_bio:
            if not any(s in domain for s in SOCIAL_DOMAINS + LINK_IN_BIO_SERVICES):
                return {"has_website": True, "url_found": domain, "method": "bio_text"}

    # 3) Domain probing — SSRF-safe, with capped timeouts (Fix 3)
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', username.lower())
    candidate_domains = [
        f"{clean_name}.com",
        f"{clean_name}.ng",
        f"{clean_name}.co",
        f"{clean_name}.store",
    ]
    if "." in username:
        candidate_domains.insert(0, username.lower())

    for domain in candidate_domains:
        for scheme in ("https", "http"):
            probe_url = f"{scheme}://{domain}"
            if not _is_safe_url(probe_url):
                continue
            try:
                resp = await asyncio.wait_for(
                    client.head(probe_url, timeout=6),
                    timeout=8,
                )
                if resp.status_code < 400:
                    return {"has_website": True, "url_found": probe_url, "method": "domain_probe"}
            except Exception:
                pass

    return {"has_website": False, "url_found": "", "method": "none_found"}


async def qualify_profile(profile: dict, config: SwarmConfig) -> dict | None:
    """Ask LLM if this profile is worth reaching out to.

    Fix 8: uses _extract_json for robust parsing.
    Fix 9: retries once on transient failure.
    """
    user_msg = json.dumps(profile, indent=2)

    for attempt in range(2):
        raw = await llm_call(config, QUALIFY_SYSTEM, user_msg, max_tokens=300, temp=0.3)
        if not raw:
            if attempt == 0:
                await asyncio.sleep(2)
                continue
            return None

        result = _extract_json(raw)
        if isinstance(result, dict):
            return result

        log.warning(f"Failed to parse qualification (attempt {attempt + 1}): {raw[:100]}")
        if attempt == 0:
            await asyncio.sleep(1)

    return None


async def generate_dm(profile: dict, qualification: dict, config: SwarmConfig) -> str | None:
    """Generate a personalized DM for a qualified lead."""
    user_msg = (
        f"Profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"Qualification:\n{json.dumps(qualification, indent=2)}"
    )
    dm = await llm_call(config, DM_SYSTEM, user_msg, max_tokens=250, temp=0.8)
    if not dm or dm.upper() == "SKIP" or len(dm) < 20:
        return None

    # Strip quotes
    if dm.startswith('"') and dm.endswith('"'):
        dm = dm[1:-1]

    return dm[:1000]  # IG DM limit


async def generate_comment(
    caption: str, username: str, config: SwarmConfig,
    media_type: str = "", thumbnail_url: str = "",
    stats: dict | None = None,
) -> str | None:
    """Generate a comment with optional image analysis via vision API.

    Fix 15: passes stats to describe_image for vision failure tracking.
    """
    # Build context — try caption first, only call vision if caption is too thin
    parts = [f"Username: @{username}"]
    if media_type:
        parts.append(f"Media type: {media_type}")

    has_caption = caption and len(caption.strip()) >= 10
    if has_caption:
        parts.append(f"Caption: {caption[:500]}")

    # Attempt 1: generate comment from caption alone (fast, no vision cost)
    comment = None
    if has_caption:
        user_msg = "\n".join(parts)
        comment = await llm_call(config, COMMENT_SYSTEM, user_msg, max_tokens=100, temp=0.8)
        # Check if LLM gave a usable comment
        if comment:
            cleaned = comment.strip()
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1].strip()
            if cleaned.upper() != "SKIP" and len(cleaned) >= 3:
                return cleaned[:300]
            log.info(f"Caption-only comment was '{cleaned}' for @{username}, trying vision...")

    # Attempt 2: enrich with vision if caption alone wasn't enough
    image_desc = None
    if thumbnail_url:
        remaining = _ig.time_remaining()
        if remaining is not None and remaining < 120:
            log.info(f"Skipping vision for @{username} — only {remaining:.0f}s left")
        else:
            image_desc = await describe_image(thumbnail_url, stats=stats)

    if image_desc:
        parts.append(f"Image description: {image_desc}")

    # If we have neither caption nor image description, give up
    if not has_caption and not image_desc:
        return None

    # Only re-call LLM if we got new image context (otherwise we already tried above)
    if not image_desc:
        return None  # caption-only already failed above

    user_msg = "\n".join(parts)
    comment = await llm_call(config, COMMENT_SYSTEM, user_msg, max_tokens=100, temp=0.8)
    if not comment:
        log.debug(f"Comment LLM returned None for @{username}")
        return None
    # Strip wrapping quotes
    cleaned = comment.strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    # Only skip on exact SKIP response, not on comments that happen to contain the word
    if cleaned.upper() == "SKIP" or len(cleaned) < 3:
        log.info(f"Comment LLM returned '{cleaned}' for @{username} — skipping")
        return None
    return cleaned[:300]


async def select_targets(
    candidates: list[dict], config: SwarmConfig,
    like_budget: int = 10, comment_budget: int = 5,
) -> list[dict]:
    """Use LLM to intelligently select which posts to engage with.

    Sends all candidate summaries in a single batch call.
    Returns selected candidates with 'action' field (ENGAGE or LIKE).
    Falls back to limit constraints if LLM fails or pool is tiny.
    """
    # Skip LLM for tiny pools — just return constrained fallback
    if len(candidates) <= 3:
        return generate_fallback(candidates, like_budget, comment_budget)

    summaries = []
    for i, c in enumerate(candidates):
        s: dict = {"i": i, "user": f"@{c['user']}"}
        if c.get("caption"):
            s["caption"] = c["caption"][:200]
        if c.get("media_type"):
            s["type"] = str(c["media_type"])
        summaries.append(s)

    budget_note = (
        f"\n\nBudget: select up to {like_budget} posts total. "
        f"Mark the best {comment_budget} as ENGAGE, rest as LIKE or SKIP."
    )
    user_msg = json.dumps(summaries, ensure_ascii=False) + budget_note

    raw = await llm_call(config, SELECT_SYSTEM, user_msg, max_tokens=800, temp=0.3)
    if not raw:
        log.warning("LLM selection failed (empty response) — falling back to limit constraints")
        return generate_fallback(candidates, like_budget, comment_budget)

    selections = _extract_json(raw, expect_array=True)
    if not isinstance(selections, list):
        log.warning(f"LLM selection parse failed: {raw[:100]} — falling back to limits")
        return generate_fallback(candidates, like_budget, comment_budget)

    selected: list[dict] = []
    engage_count = 0
    like_count = 0
    for sel in selections:
        idx = sel.get("i", -1)
        action = sel.get("a", "SKIP").upper()
        if not (0 <= idx < len(candidates)):
            continue
        total_selected = engage_count + like_count
        if action == "ENGAGE":
            if engage_count < comment_budget and total_selected < like_budget:
                selected.append(dict(candidates[idx], action="ENGAGE"))
                engage_count += 1
            elif total_selected < like_budget:
                selected.append(dict(candidates[idx], action="LIKE"))
                like_count += 1
        elif action == "LIKE":
            if total_selected < like_budget:
                selected.append(dict(candidates[idx], action="LIKE"))
                like_count += 1

    if not selected:
        log.warning("LLM selected nothing — falling back to constraints")
        return generate_fallback(candidates, like_budget, comment_budget)

    return selected


async def discover_hashtags(config: SwarmConfig) -> list[str]:
    """Use AI to generate smart hashtags for finding potential customers.

    Fix 8: uses _extract_json for robust parsing.
    """
    user_msg = (
        f"Generate 8 Instagram hashtags for finding potential BlakStudio customers. "
        f"Current date: {datetime.date.today()}. "
        f"Mix industries: real estate, fashion, food, beauty, events, tech startups, vendors. "
        f"Focus on Nigerian/African small businesses that need websites, branding, or digital presence."
    )
    raw = await llm_call(config, HASHTAG_SYSTEM, user_msg, max_tokens=200, temp=0.9)
    if not raw:
        return ["lagosvendor", "abujabusiness", "nigerianbusiness", "lagosentrepreneur"]

    tags = _extract_json(raw, expect_array=True)
    if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
        cleaned = [t.strip().replace("#", "").lower() for t in tags if t.strip()]
        return cleaned[:8]

    return ["lagosvendor", "abujabusiness", "nigerianbusiness", "lagosentrepreneur"]


# ── Rich UI helpers ───────────────────────────────────────────────

def make_banner():
    """Create the startup banner."""
    banner_text = Text()
    banner_text.append("  ██████╗ ██╗      █████╗ ██╗  ██╗\n", style="bold cyan")
    banner_text.append("  ██╔══██╗██║     ██╔══██╗██║ ██╔╝\n", style="bold cyan")
    banner_text.append("  ██████╔╝██║     ███████║█████╔╝ \n", style="bold magenta")
    banner_text.append("  ██╔══██╗██║     ██╔══██║██╔═██╗ \n", style="bold magenta")
    banner_text.append("  ██████╔╝███████╗██║  ██║██║  ██╗\n", style="bold blue")
    banner_text.append("  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝\n", style="bold blue")
    banner_text.append("  S T U D I O . D E V", style="bold white")
    return Panel(
        Align.center(banner_text),
        title="[bold yellow]Instagram Outreach Bot[/bold yellow]",
        subtitle="[dim]Powered by DeepSeek + OpenRouter/NVIDIA Vision[/dim]",
        border_style="bright_cyan",
        box=box.DOUBLE_EDGE,
        padding=(1, 2),
    )


def make_config_table(
    username: str, once: bool, interval: int, no_comment: bool,
    dry_run: bool, hashtags: list[str] | None = None, do_follow: bool = False,
    duration: str = "", no_dm: bool = False,
):
    """Create config display table."""
    table = Table(box=box.ROUNDED, border_style="dim cyan", show_header=False, padding=(0, 2))
    table.add_column("Key", style="bold white", width=15)
    table.add_column("Value", style="green")
    table.add_row("Account", f"@{username}")
    if duration:
        table.add_row("Duration", f"[bold cyan]{duration}[/bold cyan]")
        table.add_row("Mode", "[bold cyan]Timed session[/bold cyan]")
    else:
        table.add_row("Mode", "One-shot" if once else f"Loop every {interval}s")
    table.add_row("Commenting", "[red]OFF[/red]" if no_comment else "[green]ON[/green] (AI + Vision)")
    table.add_row("DM Mode", "[red]OFF[/red]" if no_dm else ("[yellow]DRY RUN[/yellow]" if dry_run else "[green]LIVE[/green]"))
    table.add_row("Follow", "[green]ON[/green]" if do_follow else "[red]OFF[/red]")
    or_count = len(OPENROUTER_MODELS) * len(OPENROUTER_API_KEYS)
    nv_count = len(NVIDIA_VISION_MODELS) * len(NVIDIA_API_KEYS)
    table.add_row("API Keys", f"[green]{len(OPENROUTER_API_KEYS)} OR + {len(NVIDIA_API_KEYS)} NV[/green]")
    table.add_row("Vision", f"[green]{or_count + nv_count} slot(s)[/green] ({len(VISION_SLOTS)} combos)")
    if hashtags:
        table.add_row("Hashtags", f"[cyan]#{', #'.join(hashtags)}[/cyan]")
    else:
        table.add_row("Hashtags", "[dim]none (feed only)[/dim]")
    table.add_row("Services", "[link=https://blakstudio.dev]blakstudio.dev[/link]")
    return table


def make_stats_table(stats: dict, total: dict, loop_count: int):
    """Create a stats summary table."""
    table = Table(
        title=f"[bold]Pass #{loop_count} Results[/bold]",
        box=box.ROUNDED,
        border_style="bright_green",
        show_lines=True,
    )
    table.add_column("Metric", style="bold", width=12)
    table.add_column("This Pass", justify="center", style="cyan", width=10)
    table.add_column("Total", justify="center", style="bold green", width=10)

    rows = [
        ("Posts", "posts"), ("Liked", "liked"), ("Commented", "commented"),
        ("Followed", "followed"), ("Qualified", "qualified"), ("DMs Sent", "dm_sent"),
        ("DM Retried", "dm_retried"), ("DM Drafted", "dm_drafted"), ("DM Skipped", "dm_skipped"),
        ("Rate Limited", "rate_limited"),
        ("Skipped", "skipped"), ("Failed", "failed"),
        ("Vision Fail", "vision_failures"),
    ]
    for label, key in rows:
        table.add_row(label, str(stats.get(key, 0)), str(total.get(key, 0)))
    return table


def make_budget_table() -> Table:
    """Create a rate-limit budget dashboard showing action capacity.

    This is the bot's 'consciousness' — it visually knows how close
    it is to each Instagram limit and plans accordingly.
    """
    has_session = bool(_ig._session_budget)
    title = "[bold]🧠 Rate-Limit Awareness[/bold]"
    if has_session:
        remaining = _ig.time_remaining()
        if remaining is not None:
            title += f" [dim]({remaining / 60:.0f} min remaining)[/dim]"

    table = Table(
        title=title,
        box=box.ROUNDED,
        border_style="bright_blue",
        show_lines=True,
    )
    table.add_column("Action", style="bold", width=10)
    table.add_column("Hour", justify="center", width=12)
    if has_session:
        table.add_column("Session", justify="center", width=12)
    table.add_column("Day", justify="center", width=12)
    table.add_column("Status", justify="center", width=18)
    table.add_column("Safe Left", justify="center", width=10)

    status = _ig.status_summary()
    for action, info in status.items():
        h_pct = info["hourly_pct"]
        d_pct = info["daily_pct"]
        worst = max(h_pct, d_pct)

        # Color-coded utilization bars
        def _bar(used, limit, pct):
            filled = int(pct / 10)
            bar = "█" * filled + "░" * (10 - filled)
            if pct >= 95:
                return f"[bold red]{bar} {used}/{limit}[/bold red]"
            elif pct >= 85:
                return f"[red]{bar} {used}/{limit}[/red]"
            elif pct >= 70:
                return f"[yellow]{bar} {used}/{limit}[/yellow]"
            else:
                return f"[green]{bar} {used}/{limit}[/green]"

        h_bar = _bar(info["hourly_used"], info["hourly_limit"], h_pct)
        d_bar = _bar(info["daily_used"], info["daily_limit"], d_pct)

        # Session budget column
        sess_bar = ""
        if has_session and info.get("session_budget") is not None:
            sb = info["session_budget"]
            su = info["session_used"] or 0
            s_pct = (su / sb * 100) if sb > 0 else 0
            sess_bar = _bar(su, sb, s_pct)

        if info["forced_pause_min"] > 0:
            status_str = f"[bold red]⏸ {info['forced_pause_min']:.0f}m[/bold red]"
        elif info.get("session_budget") and (info.get("session_used", 0) or 0) >= info["session_budget"]:
            status_str = "[bold yellow]📊 BUDGET DONE[/bold yellow]"
        elif worst >= 95:
            status_str = "[bold red]🛑 PAUSED[/bold red]"
        elif worst >= 85:
            status_str = "[red]⚠️  SLOWING[/red]"
        elif worst >= 70:
            status_str = "[yellow]👀 WATCHING[/yellow]"
        else:
            status_str = "[green]✅ CLEAR[/green]"

        safe = min(info["hourly_safe"], info["daily_safe"])
        safe_str = f"[green]{safe}[/green]" if safe > 3 else f"[red]{safe}[/red]"

        if has_session:
            table.add_row(action.replace("_", " ").title(), h_bar, sess_bar, d_bar, status_str, safe_str)
        else:
            table.add_row(action.replace("_", " ").title(), h_bar, d_bar, status_str, safe_str)

    # Session info footer
    session_min = _ig.session_duration_min()
    remaining = _ig.time_remaining()
    table.add_section()
    if has_session and remaining is not None:
        table.add_row(
            "[dim]Session[/dim]",
            f"[dim]{session_min:.0f}m elapsed[/dim]",
            f"[dim]{_ig._total_actions} actions[/dim]",
            "",
            f"[cyan]{remaining / 60:.0f}m left[/cyan]",
            "",
        )
    elif has_session:
        table.add_row(
            "[dim]Session[/dim]",
            f"[dim]{session_min:.0f}m[/dim]",
            f"[dim]{_ig._total_actions} actions[/dim]",
            "",
            "[dim]—[/dim]",
            "",
        )
    else:
        table.add_row(
            "[dim]Session[/dim]",
            f"[dim]{session_min:.0f}m[/dim]",
            f"[dim]{_ig._total_actions} actions[/dim]",
            "[dim]—[/dim]",
            "[dim]—[/dim]",
        )
    return table


def make_session_plan_table() -> Table:
    """Show the bot's session plan: what it intends to do and how it'll pace."""
    if not _ig._session_budget:
        return Table()  # empty if no duration set

    dur_min = (_ig._session_duration or 0) / 60
    table = Table(
        title=f"[bold]\U0001f9e0 Session Plan — {dur_min:.0f} minutes[/bold]",
        box=box.ROUNDED,
        border_style="bright_cyan",
        show_lines=True,
    )
    table.add_column("Action", style="bold", width=12)
    table.add_column("Budget", justify="center", width=8)
    table.add_column("Pace", justify="center", width=16)
    table.add_column("Rationale", width=30)

    plan = _ig.session_plan_table()
    for action, info in plan.items():
        budget = info["budget"]
        avg_gap = info["avg_gap_sec"]
        h_limit, d_limit = _ig.LIMITS.get(action, (50, 200))
        daily_cap = int(d_limit * _ig.WARN_AT)
        capped = budget >= daily_cap
        if capped:
            rationale = f"capped at 70% of {d_limit}/day limit"
        else:
            rationale = f"{h_limit}/hr × {dur_min:.0f}m ÷ 60 × 70% safety"
        if avg_gap >= 60:
            pace_str = f"1 every {avg_gap / 60:.1f} min"
        else:
            pace_str = f"1 every {avg_gap:.0f}s"

        table.add_row(
            action.replace("_", " ").title(),
            f"[bold cyan]{budget}[/bold cyan]",
            f"[green]{pace_str}[/green]",
            f"[dim]{rationale}[/dim]",
        )
    return table


# ── Shared post pipeline (Fix 1: DRY) ─────────────────────────────

async def _process_post(
    media_id: str, user: str, url: str, caption: str,
    media_type: str, thumbnail_url: str, has_liked: bool,
    bridge: InstagrapiBridge, config: SwarmConfig,
    dedup: dict, processed_users: set, stats: dict,
    do_comment: bool, do_follow: bool, dry_run: bool,
    progress_label: str, taken_at: int = 0,
    do_dm: bool = True,
    pre_comment: str | None = None,
) -> bool:
    """Shared pipeline: like + comment (paired) → qualify → verify → follow → DM.

    When pre_comment is provided (from the planning phase), the comment text
    has already been generated — no inline LLM/vision calls needed.
    Like and comment are coupled: we only like posts we can comment on
    (commentable posts first), or like-only posts if budget allows.

    Returns True if any Instagram API action was performed (for pacing).
    """
    already_liked = dedup["liked"]
    already_commented = dedup["commented"]
    already_dmed = dedup["dmed"]

    did_action = {}  # track whether we made any IG API call

    # Skip posts older than 6 months
    if taken_at:
        try:
            post_time = datetime.datetime.fromtimestamp(int(taken_at), tz=datetime.timezone.utc)
            age = datetime.datetime.now(datetime.timezone.utc) - post_time
            if age.days > MAX_POST_AGE_DAYS:
                console.print(f"  {progress_label} [dim]OLD[/dim]  @{user} — {age.days}d old, skipping")
                stats["skipped"] += 1
                return False
        except (ValueError, OSError, OverflowError):
            pass  # invalid timestamp — continue processing

    # Fix 6: skip liking posts from already-processed users
    if user in processed_users:
        stats["skipped"] += 1
        console.print(f"  {progress_label} [dim]SEEN[/dim]  @{user} — already processed this pass")
        return False
    processed_users.add(user)

    # Session time check — stop processing if duration exceeded
    if _ig.session_expired():
        return False

    # ── Like ──
    liked_this_post = False
    if media_id in already_liked or has_liked:
        console.print(f"  {progress_label} [dim]SKIP[/dim]  @{user}")
        stats["skipped"] += 1
    else:
        can_like, like_reason = _ig.should_act("like")
        if not can_like:
            console.print(
                f"  {progress_label} [yellow]⏸ LIKE PAUSED[/yellow] @{user} "
                f"— {like_reason}"
            )
            stats["skipped"] += 1
        else:
            result = await bridge.like_post(media_id=media_id)
            if "[SUCCESS]" in result:
                stats["liked"] += 1
                already_liked.add(media_id)
                _ig.record("like")
                _throttle.on_success()
                liked_this_post = True
                console.print(f"  {progress_label} [red]❤️  LIKED[/red] [bold]@{user}[/bold] — [dim]{url or media_id}[/dim]")
            elif "[SKIP]" in result:
                stats["skipped"] += 1
                console.print(f"  {progress_label} [dim]SKIP[/dim]  @{user}")
            elif "[RATELIMIT]" in result:
                stats["rate_limited"] += 1
                _ig.record_fail("like")
                _throttle.on_error()
                console.print(
                    f"  {progress_label} [bold red]⏸ LIKE RATE-LIMITED[/bold red] @{user} — "
                    f"reactive pause engaged"
                )
            else:
                stats["failed"] += 1
                _ig.record_fail("like")
                _throttle.on_error()
                console.print(f"  {progress_label} [bold red]FAIL[/bold red]  @{user} — {result}")
            if liked_this_post:
                did_action["like"] = True

    # ── Comment (using pre-generated text from planning phase) ──
    if do_comment and media_id not in already_commented and pre_comment:
        can_comment, comment_reason = _ig.should_act("comment")
        if not can_comment:
            console.print(
                f"         [yellow]⏸ COMMENT PAUSED[/yellow] — {comment_reason}"
            )
        else:
            result = await bridge.comment_post(media_id=media_id, text=pre_comment)
            if "[SUCCESS]" in result:
                stats["commented"] += 1
                already_commented.add(media_id)
                _ig.record("comment")
                _throttle.on_success()
                did_action["comment"] = True
                console.print(f"         [cyan]💬[/cyan] [italic]\"{pre_comment}\"[/italic] — [dim]{url}[/dim]")
            else:
                _ig.record_fail("comment")
                _throttle.on_error()
                console.print(f"         [dim red]✖ Comment failed[/dim red] @{user} — {result[:80]}")

    # ── Qualify + DM ──
    if not do_dm:
        return did_action

    with console.status(f"[dim]Checking @{user}...[/dim]", spinner="point"):
        profile_raw = await bridge.get_business_profile(username=user)
    if "[ERROR]" in profile_raw:
        return did_action

    profile = json.loads(profile_raw)
    user_id = profile.get("user_id", "")

    if user_id in already_dmed:
        console.print(f"         [dim]📩 Already DMed @{user}[/dim]")
        stats["dm_skipped"] += 1
        return did_action

    # AI qualification (Fix 9: retries inside qualify_profile)
    qual = await qualify_profile(profile, config)
    if not qual:
        return did_action

    qualify_decision = qual.get("qualify", "NO").upper()
    reason = qual.get("reason", "")
    services = qual.get("services_needed", [])
    priority = qual.get("priority", "low")

    if qualify_decision != "YES":
        console.print(f"         [dim red]✖ NOT QUALIFIED @{user}: {reason}[/dim red]")
        return did_action

    stats["qualified"] += 1

    # ── Website Verification ──
    external_url = profile.get("external_url", "")
    biography = profile.get("biography", "")
    full_name = profile.get("full_name", "")

    with console.status(f"[dim]Verifying website for @{user}...[/dim]", spinner="point"):
        web_check = await verify_no_website(user, full_name, biography, external_url)

    if web_check["has_website"]:
        found_url = web_check["url_found"]
        method = web_check["method"]
        console.print(
            f"         [yellow]🌐 WEBSITE FOUND[/yellow] @{user}: "
            f"[link={found_url}]{found_url}[/link] [dim](via {method})[/dim]"
        )
        for svc in ["website", "ecommerce"]:
            if svc in services:
                services.remove(svc)
        if not services:
            console.print(f"         [dim]Skipping — already has website, no other services needed[/dim]")
            return did_action
        qual["services_needed"] = services
        qual["has_website"] = found_url
        qual["reason"] = f"Has website ({found_url}) but still needs: {', '.join(services)}"
    else:
        console.print(f"         [green]✓ No website confirmed[/green] for @{user}")

    priority_colors = {"high": "bold red", "medium": "yellow", "low": "dim"}
    pcolor = priority_colors.get(priority, "white")
    console.print(
        f"         [green]✅ QUALIFIED[/green] [bold]@{user}[/bold] "
        f"[{pcolor}][{priority.upper()}][/{pcolor}] — needs: [cyan]{', '.join(services)}[/cyan]"
    )
    console.print(f"         [dim]{reason}[/dim]")

    # ── Follow qualified account ──
    if do_follow and user_id:
        can_follow, follow_reason = _ig.should_act("follow")
        if not can_follow:
            console.print(
                f"         [yellow]⏸ FOLLOW PAUSED[/yellow] — {follow_reason}"
            )
        else:
            follow_result = await bridge.follow_user(user_id=user_id)
            if "[SUCCESS]" in follow_result:
                stats["followed"] += 1
                _ig.record("follow")
                _throttle.on_success()
                did_action["follow"] = True
                console.print(f"         [magenta]👤 FOLLOWED[/magenta] @{user}")
            else:
                _ig.record_fail("follow")

    # ── Generate personalized DM ──
    # Proactive check: can we DM right now?
    can_dm, dm_reason = _ig.should_act("dm")
    if not can_dm:
        console.print(
            f"         [yellow]⏸ DM PAUSED[/yellow] @{user} — {dm_reason}"
        )
        # Still generate the DM and queue it for retry later
        dm_text = await generate_dm(profile, qual, config)
        if dm_text and not dry_run:
            _enqueue_dm_retry(user_id, user, dm_text)
            console.print(f"         [dim]📋 Queued DM for retry later[/dim]")
        return did_action

    dm_text = await generate_dm(profile, qual, config)
    if not dm_text:
        console.print(f"         [dim]⏭️  Skipped DM (generator returned skip)[/dim]")
        return did_action

    if dry_run:
        console.print(Panel(
            dm_text,
            title=f"[yellow]DRY RUN DM → @{user}[/yellow]",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(0, 1),
        ))
        stats["dm_drafted"] += 1  # Fix 12: separate stat for dry-run
        return did_action

    # Send the DM
    result = await bridge.send_dm(user_id=user_id, text=dm_text)
    if "[SUCCESS]" in result:
        stats["dm_sent"] += 1
        _ig.record("dm")
        _throttle.on_success()
        did_action["dm"] = True
        console.print(Panel(
            dm_text,
            title=f"[green]📩 DM SENT → @{user}[/green]",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
        ))
        # Remove from retry queue if it was there
        _remove_from_retry_queue(user_id)
    elif "[SKIP]" in result:
        stats["dm_skipped"] += 1
        console.print(f"         [dim]📩 Already DMed @{user}[/dim]")
    elif "[RATELIMIT]" in result:
        stats["failed"] += 1
        _ig.record_fail("dm")
        _throttle.on_error()
        console.print(
            f"         [bold red]❌ DM RATE-LIMITED @{user}[/bold red] — "
            f"reactive pause engaged"
        )
        # Queue for retry
        _enqueue_dm_retry(user_id, user, dm_text)
        console.print(f"         [dim]📋 Queued DM for retry later[/dim]")
    else:
        stats["failed"] += 1
        _throttle.on_error()
        console.print(f"         [bold red]❌ DM FAILED @{user}: {result}[/bold red]")
        # Queue non-rate-limit failures too (could be transient)
        _enqueue_dm_retry(user_id, user, dm_text)

    return did_action


# ── Main outreach loop ────────────────────────────────────────────

async def process_feed(
    bridge: InstagrapiBridge,
    config: SwarmConfig,
    dedup: dict,
    do_comment: bool = True,
    dry_run: bool = False,
    do_follow: bool = False,
    do_dm: bool = True,
) -> dict:
    """Four-phase pipeline: SCRAPE → SELECT → PREPARE → EXECUTE (all timed).

    Phase 1 scrapes all available posts from the paginated feed.
    Phase 2 uses the LLM to intelligently select which posts to engage with.
    Phase 3 generates comments for selected ENGAGE posts.
    Phase 4 executes with dynamic pacing based on queue size and remaining time.
    """
    stats = _new_stats()
    
    # ── Check if all requested budgets are exhausted before doing anything ──
    actions_to_check = ["like"]
    if do_comment: actions_to_check.append("comment")
    if do_dm: actions_to_check.append("dm")
    if do_follow: actions_to_check.append("follow")
    
    if _ig.all_budgets_exhausted(actions_to_check):
        console.print("[dim]  Skipping Feed scrape — all session budgets exhausted[/dim]")
        return stats

    already_liked = dedup["liked"]
    already_commented = dedup["commented"]
    t_total = time.perf_counter()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1 — SCRAPE: fetch all available posts
    # ═══════════════════════════════════════════════════════════════
    t_phase = time.perf_counter()
    console.print(Rule("[bold]Phase 1 — Scraping feed[/bold]", style="cyan"))
    console.print()

    # Scale fetch amount based on session duration so we don't over-scrape
    amount = "100" if do_comment else "60"
    if _ig._session_duration is not None:
        if _ig._session_duration <= 600:
            amount = "40"  # ~10m session
        elif _ig._session_duration <= 1800:
            amount = "60"  # ~30m session

    with console.status(f"[bold cyan]Fetching {amount} timeline posts...", spinner="dots"):
        feed_raw = await bridge.get_timeline_feed(amount=amount)
    feed_data = json.loads(feed_raw)
    posts = feed_data.get("posts", [])

    if not posts:
        console.print("[yellow]No posts found on timeline.[/yellow]")
        return stats

    stats["posts"] = len(posts)

    # Basic filtering — fast, no API/LLM calls
    seen_users: set[str] = set()
    raw_candidates: list[dict] = []
    filter_stats = {"duped": 0, "already_liked": 0, "old": 0, "invalid": 0}

    for post in posts:
        media_id = str(post.get("id", ""))
        user = post.get("user", "unknown")
        url = post.get("url", "")
        caption = post.get("caption", "") or ""
        has_liked = post.get("has_liked", False)
        media_type = post.get("media_type", "")
        thumbnail_url = post.get("thumbnail_url", "")
        taken_at = post.get("taken_at", 0)

        if not media_id or not user:
            filter_stats["invalid"] += 1
            continue
        if user in seen_users:
            filter_stats["duped"] += 1
            continue
        seen_users.add(user)
        if media_id in already_liked or has_liked:
            filter_stats["already_liked"] += 1
            stats["skipped"] += 1
            continue
        if taken_at:
            try:
                post_time = datetime.datetime.fromtimestamp(int(taken_at), tz=datetime.timezone.utc)
                age = datetime.datetime.now(datetime.timezone.utc) - post_time
                if age.days > MAX_POST_AGE_DAYS:
                    filter_stats["old"] += 1
                    stats["skipped"] += 1
                    continue
            except (ValueError, OSError, OverflowError):
                pass

        raw_candidates.append({
            "media_id": media_id, "user": user, "url": url,
            "caption": caption, "media_type": media_type,
            "thumbnail_url": thumbnail_url, "has_liked": has_liked,
            "taken_at": taken_at,
        })

    scrape_elapsed = time.perf_counter() - t_phase
    filtered_parts = []
    if filter_stats["already_liked"]:
        filtered_parts.append(f"{filter_stats['already_liked']} already liked")
    if filter_stats["duped"]:
        filtered_parts.append(f"{filter_stats['duped']} dupes")
    if filter_stats["old"]:
        filtered_parts.append(f"{filter_stats['old']} old")
    filter_info = f"\n  [dim]Filtered: {', '.join(filtered_parts)}[/dim]" if filtered_parts else ""
    console.print(
        f"  [green]Scraped {len(posts)} posts → {len(raw_candidates)} candidates[/green] "
        f"[dim]({_fmt_elapsed(scrape_elapsed)})[/dim]{filter_info}"
    )
    console.print()

    if not raw_candidates:
        console.print("[yellow]No actionable posts after filtering.[/yellow]")
        return stats

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2 — SELECT: LLM picks which posts to engage with
    # ═══════════════════════════════════════════════════════════════
    t_phase = time.perf_counter()
    console.print(Rule("[bold]Phase 2 — AI selecting targets[/bold]", style="yellow"))
    console.print()

    # Get remaining budget from rate limiter
    if _ig._session_budget:
        like_budget = _ig._session_budget.get("like", 10)
        like_used = _ig.count_in_window("like", _ig._session_duration or 3600)
        like_remaining = max(1, like_budget - like_used)
        comment_budget = _ig._session_budget.get("comment", 5)
        comment_used = _ig.count_in_window("comment", _ig._session_duration or 3600)
        comment_remaining = max(1, comment_budget - comment_used) if do_comment else 0
    else:
        like_remaining = 15
        comment_remaining = 8 if do_comment else 0

    with console.status("[bold yellow]AI analyzing candidates...", spinner="dots"):
        selected = await select_targets(
            raw_candidates, config,
            like_budget=like_remaining,
            comment_budget=comment_remaining,
        )

    engage_count = sum(1 for s in selected if s.get("action") == "ENGAGE")
    like_count = sum(1 for s in selected if s.get("action") == "LIKE")
    select_elapsed = time.perf_counter() - t_phase
    console.print(
        f"  [green]AI selected {len(selected)} targets: "
        f"{engage_count} engage, {like_count} like-only[/green] "
        f"[dim]({_fmt_elapsed(select_elapsed)})[/dim]"
    )
    console.print()

    if not selected:
        console.print("[yellow]AI found no suitable targets.[/yellow]")
        return stats

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3 — PREPARE: generate comments for ENGAGE targets
    # ═══════════════════════════════════════════════════════════════
    engage_targets = [s for s in selected if s["action"] == "ENGAGE"]
    if do_comment and engage_targets:
        t_phase = time.perf_counter()
        console.print(Rule(
            f"[bold]Phase 3 — Generating {len(engage_targets)} comments[/bold]",
            style="cyan",
        ))
        console.print()

        for item in engage_targets:
            if item["media_id"] in already_commented:
                continue
            comment = await generate_comment(
                item["caption"], item["user"], config,
                media_type=item["media_type"],
                thumbnail_url=item["thumbnail_url"],
                stats=stats,
            )
            item["comment"] = comment

        commentable = sum(1 for t in engage_targets if t.get("comment"))
        prep_elapsed = time.perf_counter() - t_phase
        console.print(
            f"  [green]Generated {commentable}/{len(engage_targets)} comments[/green] "
            f"[dim]({_fmt_elapsed(prep_elapsed)})[/dim]"
        )
        console.print()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4 — EXECUTE: like + comment with dynamic pacing
    # ═══════════════════════════════════════════════════════════════
    with_comments = [s for s in selected if s.get("comment")]
    without_comments = [s for s in selected if not s.get("comment")]
    work_queue = with_comments + without_comments

    if not work_queue:
        console.print("[yellow]No actionable posts found.[/yellow]")
        return stats

    # Reset session clock — scrape/select/prepare time doesn't count
    prep_elapsed = time.perf_counter() - t_total
    _ig.adjust_session_clock(prep_elapsed)
    console.print(
        f"  [dim]Prep took {_fmt_elapsed(prep_elapsed)} — session clock "
        f"shifted, full duration preserved for execution[/dim]"
    )
    console.print()

    t_execute = time.perf_counter()
    phase_num = 4 if (do_comment and engage_targets) else 3
    console.print(Rule(
        f"[bold]Phase {phase_num} — Executing on {len(work_queue)} posts[/bold]",
        style="green",
    ))
    console.print()

    # Show dynamic pacing plan
    remaining = _ig.time_remaining()
    if remaining is not None:
        delay = _ig.get_delay("like")
        console.print(
            f"  [dim]⏱ Base pacing: ~{_fmt_elapsed(delay)} between posts "
            f"(target: {len(work_queue)} posts)[/dim]"
        )
        console.print()

    processed_users: set[str] = set()
    for i, item in enumerate(work_queue, 1):
        if _ig.session_expired():
            console.print("  [yellow]⏱️  Session expired — stopping[/yellow]")
            break

        t_post = time.perf_counter()
        progress_label = f"[dim][{i}/{len(work_queue)}][/dim]"

        did_action = await _process_post(
            media_id=item["media_id"], user=item["user"],
            url=item["url"], caption=item["caption"],
            media_type=item["media_type"],
            thumbnail_url=item["thumbnail_url"],
            has_liked=item["has_liked"],
            bridge=bridge, config=config,
            dedup=dedup, processed_users=processed_users, stats=stats,
            do_comment=do_comment, do_follow=do_follow,
            dry_run=dry_run, progress_label=progress_label,
            taken_at=item["taken_at"], do_dm=do_dm,
            pre_comment=item.get("comment"),
        )

        post_elapsed = time.perf_counter() - t_post
        if did_action:
            console.print(f"         [dim]⏱ {_fmt_elapsed(post_elapsed)}[/dim]")

        delay = _ig.get_delay("like")
        if isinstance(did_action, dict):
            if did_action.get("dm"):
                delay = _ig.get_delay("dm")
            elif did_action.get("comment"):
                delay = _ig.get_delay("comment")
            elif did_action.get("follow"):
                delay = _ig.get_delay("follow")
        
        delay = delay - post_elapsed
        delay = max(delay, 5.0)  # always at least 5s
        await asyncio.sleep(delay)

    execute_elapsed = time.perf_counter() - t_execute
    total_elapsed = time.perf_counter() - t_total
    console.print()
    console.print(
        f"  [dim]Execute: {_fmt_elapsed(execute_elapsed)} | "
        f"Total pass: {_fmt_elapsed(total_elapsed)}[/dim]"
    )

    return stats


# ── Hashtag discovery ─────────────────────────────────────────────

async def process_hashtags(
    bridge: InstagrapiBridge,
    config: SwarmConfig,
    hashtags: list[str],
    dedup: dict,
    do_comment: bool = True,
    dry_run: bool = False,
    do_follow: bool = False,
    do_dm: bool = True,
) -> dict:
    """Hashtag processing: SCRAPE → SELECT → PREPARE → EXECUTE per tag (all timed)."""
    stats = _new_stats()
    
    # ── Check if all requested budgets are exhausted before doing anything ──
    actions_to_check = ["like"]
    if do_comment: actions_to_check.append("comment")
    if do_dm: actions_to_check.append("dm")
    if do_follow: actions_to_check.append("follow")
    
    if _ig.all_budgets_exhausted(actions_to_check):
        console.print("[dim]  Skipping Hashtags scrape — all session budgets exhausted[/dim]")
        return stats

    already_liked = dedup["liked"]
    already_commented = dedup["commented"]
    t_total = time.perf_counter()

    for tag in hashtags:
        tag = tag.strip().replace("#", "")
        if not tag:
            continue

        console.print(Rule(f"[bold magenta]#{tag}[/bold magenta]", style="magenta"))

        t_tag_start = time.perf_counter()

        # ── Phase 1: Scrape ──
        t_phase = time.perf_counter()

        # Scale tag fetch amount based on session duration
        tag_amount = "50"
        if _ig._session_duration is not None:
            if _ig._session_duration <= 600:
                tag_amount = "25"  # ~10m session
            elif _ig._session_duration <= 1800:
                tag_amount = "40"  # ~30m session

        with console.status(f"[bold cyan]Scraping #{tag} ({tag_amount} posts)...", spinner="dots"):
            feed_raw = await bridge.hashtag_feed(hashtag=tag, amount=tag_amount)

        if "[ERROR]" in feed_raw:
            console.print(f"  [red]Failed to fetch #{tag}: {feed_raw}[/red]")
            continue

        posts = json.loads(feed_raw)
        if not isinstance(posts, list):
            posts = posts.get("posts", []) if isinstance(posts, dict) else []

        if not posts:
            console.print(f"  [yellow]No posts found for #{tag}[/yellow]")
            continue

        stats["posts"] += len(posts)

        # Filter
        seen_users: set[str] = set()
        raw_candidates: list[dict] = []

        for post in posts:
            media_id = str(post.get("id", post.get("pk", "")))
            user_data = post.get("user", {})
            user = user_data.get("username", "") if isinstance(user_data, dict) else str(user_data)
            code = post.get("code", "")
            caption = post.get("caption_text", post.get("caption", "")) or ""
            url = f"https://www.instagram.com/p/{code}/" if code else ""
            media_type = post.get("media_type", "")
            thumbnail_url = post.get("thumbnail_url", "")
            if not thumbnail_url:
                candidates_img = post.get("image_versions2", {}).get("candidates", [])
                if candidates_img:
                    thumbnail_url = candidates_img[0].get("url", "")
            taken_at = post.get("taken_at", 0)

            if not media_id or not user:
                continue
            if user in seen_users:
                continue
            seen_users.add(user)
            if media_id in already_liked:
                stats["skipped"] += 1
                continue

            raw_candidates.append({
                "media_id": media_id, "user": user, "url": url,
                "caption": caption, "media_type": media_type,
                "thumbnail_url": thumbnail_url, "has_liked": False,
                "taken_at": taken_at,
            })

        scrape_elapsed = time.perf_counter() - t_phase
        console.print(
            f"  [green]Scraped {len(posts)} posts → {len(raw_candidates)} candidates[/green] "
            f"[dim]({_fmt_elapsed(scrape_elapsed)})[/dim]"
        )

        if not raw_candidates:
            continue

        # ── Phase 2: Select ──
        t_phase = time.perf_counter()
        if _ig._session_budget:
            like_budget = _ig._session_budget.get("like", 10)
            like_used = _ig.count_in_window("like", _ig._session_duration or 3600)
            like_remaining = max(1, like_budget - like_used)
            comment_budget = _ig._session_budget.get("comment", 5)
            comment_used = _ig.count_in_window("comment", _ig._session_duration or 3600)
            comment_remaining = max(1, comment_budget - comment_used) if do_comment else 0
        else:
            like_remaining = 10
            comment_remaining = 5 if do_comment else 0

        with console.status(f"[bold yellow]AI selecting #{tag} targets...", spinner="dots"):
            selected = await select_targets(
                raw_candidates, config,
                like_budget=like_remaining,
                comment_budget=comment_remaining,
            )

        engage_count = sum(1 for s in selected if s.get("action") == "ENGAGE")
        like_count = sum(1 for s in selected if s.get("action") == "LIKE")
        select_elapsed = time.perf_counter() - t_phase
        console.print(
            f"  [green]AI: {len(selected)} targets — {engage_count} engage, "
            f"{like_count} like-only[/green] "
            f"[dim]({_fmt_elapsed(select_elapsed)})[/dim]"
        )

        if not selected:
            continue

        # ── Phase 3: Prepare comments ──
        engage_targets = [s for s in selected if s["action"] == "ENGAGE"]
        if do_comment and engage_targets:
            t_phase = time.perf_counter()
            for item in engage_targets:
                if item["media_id"] in already_commented:
                    continue
                comment = await generate_comment(
                    item["caption"], item["user"], config,
                    media_type=item["media_type"],
                    thumbnail_url=item["thumbnail_url"],
                    stats=stats,
                )
                item["comment"] = comment

            commentable = sum(1 for t in engage_targets if t.get("comment"))
            prep_elapsed = time.perf_counter() - t_phase
            console.print(
                f"  [green]Comments: {commentable}/{len(engage_targets)}[/green] "
                f"[dim]({_fmt_elapsed(prep_elapsed)})[/dim]"
            )

        # ── Phase 4: Execute ──
        with_comments = [s for s in selected if s.get("comment")]
        without_comments = [s for s in selected if not s.get("comment")]
        work_queue = with_comments + without_comments
        console.print()

        # Reset session clock — scrape/select/prepare time excluded
        prep_elapsed = time.perf_counter() - t_tag_start
        _ig.adjust_session_clock(prep_elapsed)

        t_execute = time.perf_counter()
        processed_users: set[str] = set()
        for i, item in enumerate(work_queue, 1):
            if _ig.session_expired():
                break

            t_post = time.perf_counter()
            progress_label = f"[dim][#{tag} {i}/{len(work_queue)}][/dim]"

            did_action = await _process_post(
                media_id=item["media_id"], user=item["user"],
                url=item["url"], caption=item["caption"],
                media_type=item["media_type"],
                thumbnail_url=item["thumbnail_url"],
                has_liked=item["has_liked"],
                bridge=bridge, config=config,
                dedup=dedup, processed_users=processed_users, stats=stats,
                do_comment=do_comment, do_follow=do_follow,
                dry_run=dry_run, progress_label=progress_label,
                taken_at=item["taken_at"], do_dm=do_dm,
                pre_comment=item.get("comment"),
            )

            post_elapsed = time.perf_counter() - t_post
            if did_action:
                console.print(f"         [dim]⏱ {_fmt_elapsed(post_elapsed)}[/dim]")

        delay = _ig.get_delay("like")
        if isinstance(did_action, dict):
            if did_action.get("dm"):
                delay = _ig.get_delay("dm")
            elif did_action.get("comment"):
                delay = _ig.get_delay("comment")
            elif did_action.get("follow"):
                delay = _ig.get_delay("follow")
        
        delay = delay - post_elapsed
        delay = max(delay, 5.0)  # always at least 5s
        await asyncio.sleep(delay)

        execute_elapsed = time.perf_counter() - t_execute
        console.print(f"  [dim]#{tag} execute: {_fmt_elapsed(execute_elapsed)}[/dim]")

    total_elapsed = time.perf_counter() - t_total
    console.print(f"  [dim]Hashtags total: {_fmt_elapsed(total_elapsed)}[/dim]")

    return stats


# ── Entrypoint (Fix 10: argparse, Fix 11: specific exception handling) ─

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BlakStudio.dev — Intelligent Instagram DM Outreach Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=600, help="Loop interval in seconds (default: 600)")
    parser.add_argument(
        "--duration", type=str, default="",
        help="Session duration e.g. '5m', '30m', '1h', '2h30m'. "
             "Bot calculates optimal action pacing to fill this window safely.",
    )
    parser.add_argument("--no-comment", action="store_true", help="Disable auto-commenting")
    parser.add_argument("--dry-run", action="store_true", help="Generate DMs but don't send")
    parser.add_argument("--no-dm", action="store_true", help="Disable qualifying and DM outreach (like/comment/follow only)")
    parser.add_argument("--no-follow", action="store_true", help="Disable auto-follow")
    parser.add_argument("--no-hashtags", action="store_true", help="Disable hashtag discovery")
    parser.add_argument(
        "--hashtags", type=str, default="",
        help="Comma-separated hashtags to search (e.g., 'smallbusiness,lagosvendor')",
    )
    return parser.parse_args()


async def main():
    args = _parse_args()

    username = os.environ.get("IG_USERNAME", "")
    password = os.environ.get("IG_PASSWORD", "")

    if not username or not password:
        username = input("Instagram username: ").strip()
        password = input("Instagram password: ").strip()

    do_follow = not args.no_follow
    do_dm = not args.no_dm
    manual_hashtags = [
        t.strip().replace("#", "") for t in args.hashtags.split(",") if t.strip()
    ] if args.hashtags else []

    # ── Beautiful startup ──
    console.print()
    console.print(make_banner())
    console.print()
    console.print(Align.center(make_config_table(
        username, args.once, args.interval, args.no_comment, args.dry_run,
        hashtags=(manual_hashtags or ["AI auto-discover"]) if not args.no_hashtags else None,
        do_follow=do_follow,
        duration=args.duration,
        no_dm=args.no_dm,
    )))
    if VISION_SLOTS:
        providers = []
        if OPENROUTER_API_KEYS:
            providers.append(f"{len(OPENROUTER_API_KEYS)} OpenRouter key(s)")
        if NVIDIA_API_KEYS:
            providers.append(f"{len(NVIDIA_API_KEYS)} NVIDIA key(s)")
        console.print(Align.center(
            f"[dim]Vision: {len(VISION_SLOTS)} slots — {', '.join(providers)}[/dim]"
        ))
    console.print()

    # Login with spinner
    with console.status("[bold cyan]Logging in to Instagram...", spinner="dots12"):
        bridge = InstagrapiBridge(username, password)
    config = SwarmConfig.from_env()
    console.print("[green]✓ Logged in successfully[/green]")
    console.print()

    # ── Session duration planning ──
    if args.duration:
        dur_sec = InstagramRateLimiter.parse_duration(args.duration)
        if dur_sec > 0:
            _ig.plan_session(dur_sec)
            console.print(Align.center(make_session_plan_table()))
            console.print()
            console.print(Align.center(
                f"[bold cyan]⏱️  Session will auto-stop in {dur_sec / 60:.0f} minutes. "
                f"Actions are paced to fill this window safely.[/bold cyan]"
            ))
            console.print()
        else:
            console.print(f"[yellow]⚠ Could not parse duration '{args.duration}' — running unlimited[/yellow]")
            console.print()

    loop_count = 0
    total = _new_stats()
    consecutive_pass_errors = 0

    while True:
        # ── Session time check ──
        if _ig.session_expired():
            remaining_sec = 0.0
            console.print()
            console.print(Rule("[bold yellow]\u23f1\ufe0f  Session Duration Reached[/bold yellow]", style="yellow"))
            console.print(Align.center(
                f"[bold yellow]Planned duration of {(_ig._session_duration or 0) / 60:.0f} min has elapsed. "
                f"Stopping gracefully.[/bold yellow]"
            ))
            console.print()
            break

        loop_count += 1
        console.print(Rule(f"[bold bright_cyan]Pass #{loop_count}[/bold bright_cyan]", style="bright_cyan"))
        console.print()

        try:
            # Fix 2: load dedup sets once per pass
            dedup = await _load_dedup_sets(bridge)

            # Sliding-window rate limiter is always current — no reset needed

            # ── Retry previously failed DMs ──
            retry_queue = _load_dm_retry_queue()
            if retry_queue and not args.dry_run and do_dm:
                console.print(Rule("[bold yellow]Retrying Failed DMs[/bold yellow]", style="yellow"))
                console.print(f"  [yellow]📋 {len(retry_queue)} DM(s) in retry queue[/yellow]")
                retried = 0
                still_failed = []
                for item in retry_queue:
                    uid = item["user_id"]
                    uname = item.get("username", uid)
                    dm_text = item["dm_text"]
                    retries = item.get("retries", 0)

                    # Skip if already DMed (e.g., manually)
                    if uid in dedup["dmed"]:
                        console.print(f"         [dim]📩 Already DMed @{uname} — removing from queue[/dim]")
                        continue

                    # Check if DMs are allowed proactively
                    can_retry_dm, retry_dm_reason = _ig.should_act("dm")
                    if not can_retry_dm:
                        console.print(
                            f"         [yellow]⏸ DMs paused — {retry_dm_reason}, "
                            f"keeping {len(retry_queue) - retried} in queue[/yellow]"
                        )
                        still_failed.extend(retry_queue[retry_queue.index(item):])
                        break

                    # Give up after 3 retries
                    if retries >= 3:
                        console.print(f"         [dim red]✖ Giving up on @{uname} after {retries} retries[/dim red]")
                        continue

                    # Attempt the DM with human-like delay
                    await asyncio.sleep(_ig.get_delay("dm"))
                    result = await bridge.send_dm(user_id=uid, text=dm_text)

                    if "[SUCCESS]" in result:
                        retried += 1
                        _ig.record("dm")
                        total["dm_sent"] += 1
                        console.print(Panel(
                            dm_text,
                            title=f"[green]📩 DM RETRY OK → @{uname}[/green]",
                            border_style="green",
                            box=box.ROUNDED,
                            padding=(0, 1),
                        ))
                    elif "[RATELIMIT]" in result:
                        _ig.record_fail("dm")
                        item["retries"] = retries + 1
                        still_failed.append(item)
                        console.print(
                            f"         [bold red]❌ DM retry rate-limited @{uname}[/bold red] — "
                            f"reactive pause engaged"
                        )
                        # Keep remaining queue items for next pass
                        idx = retry_queue.index(item)
                        for remaining_item in retry_queue[idx + 1:]:
                            if remaining_item not in still_failed:
                                still_failed.append(remaining_item)
                        break
                    elif "[SKIP]" in result:
                        console.print(f"         [dim]📩 Already DMed @{uname}[/dim]")
                    else:
                        item["retries"] = retries + 1
                        still_failed.append(item)
                        console.print(f"         [red]❌ DM retry failed @{uname}: {result}[/red]")

                _save_dm_retry_queue(still_failed)
                if retried:
                    console.print(f"  [green]✅ Retried {retried} DM(s) successfully[/green]")
                if still_failed:
                    console.print(f"  [yellow]📋 {len(still_failed)} DM(s) still in retry queue[/yellow]")
                console.print()

            # ── Timeline feed ──
            console.print(Rule("[bold]Timeline Feed[/bold]", style="cyan"))
            feed_stats = await process_feed(
                bridge, config, dedup,
                do_comment=not args.no_comment, dry_run=args.dry_run, do_follow=do_follow,
                do_dm=do_dm,
            )
            for k in total:
                total[k] += feed_stats.get(k, 0)

            # ── Hashtag discovery ──
            if not args.no_hashtags:
                console.print()
                console.print(Rule("[bold]Hashtag Discovery[/bold]", style="magenta"))

                # Use manual hashtags if provided, otherwise AI-discover
                if manual_hashtags:
                    hashtags = manual_hashtags
                    console.print(f"  [cyan]Using manual hashtags: #{', #'.join(hashtags)}[/cyan]")
                else:
                    with console.status("[bold magenta]AI discovering hashtags...", spinner="dots"):
                        hashtags = await discover_hashtags(config)
                    console.print(f"  [magenta]🧠 AI picked: [bold]#{', #'.join(hashtags)}[/bold][/magenta]")
                console.print()

                ht_stats = await process_hashtags(
                    bridge, config, hashtags, dedup,
                    do_comment=not args.no_comment, dry_run=args.dry_run, do_follow=do_follow,
                    do_dm=do_dm,
                )
                for k in total:
                    total[k] += ht_stats.get(k, 0)
                # Merge into feed_stats for the pass display
                for k in feed_stats:
                    feed_stats[k] = feed_stats.get(k, 0) + ht_stats.get(k, 0)

            console.print()
            console.print(Align.center(make_stats_table(feed_stats, total, loop_count)))
            console.print(Align.center(make_budget_table()))
            console.print()
            consecutive_pass_errors = 0  # reset on success

        # Fix 11: handle auth/connection errors specifically
        except (ConnectionError, httpx.HTTPStatusError) as e:
            consecutive_pass_errors += 1
            console.print(f"[bold red]Connection error in pass #{loop_count}: {e}[/bold red]")
            if consecutive_pass_errors >= 3:
                console.print("[bold red]3 consecutive failures — attempting re-login...[/bold red]")
                try:
                    bridge.login()
                    console.print("[green]Re-login successful[/green]")
                    consecutive_pass_errors = 0
                except Exception as login_err:
                    console.print(f"[bold red]Re-login failed: {login_err}. Exiting.[/bold red]")
                    break
        except Exception as e:
            consecutive_pass_errors += 1
            console.print(f"[bold red]Error in pass #{loop_count}: {e}[/bold red]")
            if consecutive_pass_errors >= 5:
                console.print("[bold red]Too many consecutive errors. Exiting.[/bold red]")
                break

        if args.once:
            break

        # Skip countdown if session is about to expire
        if _ig.session_expired():
            continue  # loop back to the while True check which will break

        # Countdown timer — adaptive interval based on throttle state (Fix 14)
        if _ig._session_duration:
            # Timed session: short pause between passes so we use the full window
            actual_interval = random.randint(15, 30)
        else:
            actual_interval = int(args.interval * _throttle.multiplier)
        # Cap countdown to remaining session time if duration is set
        remaining = _ig.time_remaining()
        if remaining is not None:
            actual_interval = min(actual_interval, max(1, int(remaining)))
        console.print()
        with Progress(
            SpinnerColumn("dots"),
            TextColumn("[bold blue]Next scan in"),
            BarColumn(bar_width=30, style="cyan", complete_style="bright_cyan"),
            TaskProgressColumn(),
            TextColumn("[dim]seconds[/dim]"),
            console=console,
        ) as progress:
            task = progress.add_task("countdown", total=actual_interval)
            for tick in range(actual_interval):
                if _ig.session_expired():
                    break
                await asyncio.sleep(1)
                progress.update(task, advance=1)
        console.print()

    # Cleanup shared HTTP client
    global _http_pool
    if _http_pool and not _http_pool.is_closed:
        await _http_pool.aclose()

    # Final summary
    console.print(Rule("[bold green]Session Complete[/bold green]", style="green"))
    console.print(Align.center(make_stats_table(total, total, loop_count)))
    console.print(Align.center(make_budget_table()))
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
