"""Auto-like and smart-comment on your Instagram timeline feed.

Runs in an infinite loop, checking your feed periodically.
Uses your local DeepSeek to generate relevant comments.
Session is cached in ig_settings.json — you only login once.

Usage:
    export IG_USERNAME="your_username"
    export IG_PASSWORD="your_password"

    # Run forever (like + comment every 5 minutes)
    python -m cookbook.examples.auto_like_feed

    # One-shot (single pass, then exit)
    python -m cookbook.examples.auto_like_feed --once

    # Like only, no commenting
    python -m cookbook.examples.auto_like_feed --no-comment

    # Custom interval (seconds between loops)
    python -m cookbook.examples.auto_like_feed --interval 600
"""

import asyncio
import json
import os
import sys
import time
import random
import logging
import base64

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
log = logging.getLogger("auto_feed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

# ── Vision API config ────────────────────────────────────────────
# Rate limits for :free models: 20 req/min per model, 50 req/day (free tier).
# By rotating across 5 models we get ~100 req/min and spread the daily budget.
VISION_API_BASE = os.environ.get("VISION_API_BASE", "https://openrouter.ai/api/v1")
VISION_MODELS: list[str] = [
    "google/gemma-3-27b-it:free",            # 27B, 131K ctx — best quality
    "mistralai/mistral-small-3.1-24b-instruct:free",  # 24B, 128K ctx
    "nvidia/nemotron-nano-12b-v2-vl:free",   # 12B, 128K ctx — also handles video
    "google/gemma-3-12b-it:free",            # 12B, 32K ctx
    "google/gemma-3-4b-it:free",             # 4B, 32K ctx — fastest
]
_vision_idx = 0  # round-robin index
VISION_API_KEY = os.environ.get("VISION_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")

# ── Comment generation via DeepSeek ──────────────────────────────

COMMENT_SYSTEM = """You are a friendly Instagram user. Given a post's details, write a SHORT, natural comment (1-2 sentences max).

You may receive:
- Caption text from the post
- An AI-generated description of the image/video (if available)
- The media type (photo/video/carousel)

Rules:
- Be genuine and relevant — reference something SPECIFIC from the caption or image description
- No emojis spam — 0-2 emojis max
- No generic comments like "Nice!" or "Great post!" — be specific
- Match the vibe: professional for business posts, casual for personal posts
- Never be promotional or salesy
- If the caption is in another language, comment in that language
- If the post is clearly an ad/spam, respond with SKIP
- Keep it under 150 characters
- IMPORTANT: If the post is showcasing products they are selling (shoes, clothes, jewellery, bags, food, etc.), do NOT ask about price, size, cost, or availability. Instead, compliment the aesthetics, the color, the design, or how beautiful the item looks.
- Never ask "How much?", "What size?", "Where can I get this?" — just appreciate the beauty.

Respond with ONLY the comment text, nothing else."""

VISION_DESCRIBE_PROMPT = """Describe this Instagram post image in 1-2 sentences. Focus on:
- What's shown (people, objects, scenery, food, fashion, etc.)
- The mood/vibe (professional, casual, celebratory, etc.)
- Any text overlays visible
Be concise and factual."""


async def describe_image(image_url: str) -> str | None:
    """Use OpenRouter free vision models to describe an image.

    Rotates through all 5 free models round-robin style.
    On 429 rate-limit, waits 2s and tries the next model.
    """
    global _vision_idx
    if not VISION_API_BASE or not image_url:
        return None

    headers = {
        "Authorization": f"Bearer {VISION_API_KEY}",
        "HTTP-Referer": "https://blakstudio.dev",
        "X-Title": "BlakStudio Instagram Bot",
    }
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

    for attempt in range(len(VISION_MODELS)):
        model = VISION_MODELS[(_vision_idx) % len(VISION_MODELS)]
        _vision_idx = (_vision_idx + 1) % len(VISION_MODELS)
        try:
            async with httpx.AsyncClient(base_url=VISION_API_BASE, timeout=45, headers=headers) as client:
                resp = await client.post("/chat/completions", json={"model": model, **payload})
                if resp.status_code == 429:
                    log.warning(f"Vision 429 on {model.split('/')[-1]}, rotating...")
                    await asyncio.sleep(2)
                    continue
                resp.raise_for_status()
                data = resp.json()
                desc = data["choices"][0]["message"]["content"].strip()
                log.info(f"  👁️  Vision ({model.split('/')[-1]}): {desc[:80]}...")
                return desc
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                log.warning(f"Vision 429 on {model.split('/')[-1]}, rotating...")
                await asyncio.sleep(2)
                continue
            log.warning(f"Vision failed ({model}): {e}")
            continue
        except Exception as e:
            log.warning(f"Vision failed ({model}): {e}")
            continue
    return None


async def generate_comment(
    caption: str, username: str, config: SwarmConfig,
    media_type: str = "", thumbnail_url: str = "", video_url: str = "",
) -> str | None:
    """Generate a relevant comment using caption + optional image analysis."""
    # Try vision analysis first (if a vision API is configured)
    image_desc = None
    img_to_analyze = thumbnail_url or ""
    if img_to_analyze:
        image_desc = await describe_image(img_to_analyze)

    # Build context for the text LLM
    parts = [f"Username: @{username}"]
    if media_type:
        parts.append(f"Media type: {media_type}")
    if caption and len(caption.strip()) >= 5:
        parts.append(f"Caption: {caption[:500]}")
    if image_desc:
        parts.append(f"Image description: {image_desc}")

    # If we have neither caption nor image description, skip
    if not caption and not image_desc:
        return None
    if not image_desc and (not caption or len(caption.strip()) < 10):
        return None

    user_msg = "\n".join(parts)

    api_base = config.api_base or "http://localhost:8000/v1"
    try:
        async with httpx.AsyncClient(base_url=api_base, timeout=30) as client:
            resp = await client.post("/chat/completions", json={
                "model": config.default_model,
                "messages": [
                    {"role": "system", "content": COMMENT_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.8,
                "max_tokens": 100,
                "stream": False,
            })
            resp.raise_for_status()
            data = resp.json()
            comment = data["choices"][0]["message"]["content"].strip()

            # If the LLM says skip, don't comment
            if comment.upper() == "SKIP" or len(comment) < 3:
                return None

            # Safety: strip quotes the LLM might wrap it in
            if comment.startswith('"') and comment.endswith('"'):
                comment = comment[1:-1]

            return comment[:300]  # Instagram comment limit safeguard
    except Exception as e:
        log.warning(f"Comment generation failed: {e}")
        return None


# ── Main feed processing loop ────────────────────────────────────

async def process_feed(bridge: InstagrapiBridge, config: SwarmConfig, do_comment: bool = True) -> dict:
    """Single pass: fetch feed, like new posts, comment on relevant ones."""
    stats = {"liked": 0, "skipped": 0, "commented": 0, "failed": 0}

    # Load already-processed IDs
    liked_raw = await bridge.get_liked_posts()
    liked_data = json.loads(liked_raw)
    already_liked = set(str(i) for i in liked_data.get("liked_ids", []))

    commented_raw = await bridge.get_commented_posts()
    commented_data = json.loads(commented_raw)
    already_commented = set(str(i) for i in commented_data.get("commented_ids", []))

    # Fetch feed
    with console.status("[bold cyan]Fetching timeline feed...", spinner="dots"):
        feed_raw = await bridge.get_timeline_feed(amount="30")
    feed_data = json.loads(feed_raw)
    posts = feed_data.get("posts", [])

    if not posts:
        console.print("[yellow]No posts found on timeline.[/yellow]")
        return stats

    console.print(Rule(f"[bold]Processing {len(posts)} posts[/bold]", style="cyan"))
    console.print()

    for i, post in enumerate(posts, 1):
        media_id = str(post.get("id", ""))
        user = post.get("user", "unknown")
        url = post.get("url", "")
        caption = post.get("caption", "") or ""
        has_liked = post.get("has_liked", False)
        media_type = post.get("media_type", "")
        thumbnail_url = post.get("thumbnail_url", "")
        video_url = post.get("video_url", "")

        if not media_id:
            continue

        progress_label = f"[dim][{i}/{len(posts)}][/dim]"

        # ── Like ──
        if media_id in already_liked or has_liked:
            console.print(f"  {progress_label} [dim]SKIP[/dim]  @{user}")
            stats["skipped"] += 1
        else:
            result = await bridge.like_post(media_id=media_id)
            if "[SUCCESS]" in result:
                stats["liked"] += 1
                console.print(f"  {progress_label} [red]❤️  LIKED[/red] [bold]@{user}[/bold] — [dim]{url or media_id}[/dim]")
            elif "[SKIP]" in result:
                stats["skipped"] += 1
                console.print(f"  {progress_label} [dim]SKIP[/dim]  @{user}")
            else:
                stats["failed"] += 1
                console.print(f"  {progress_label} [bold red]FAIL[/bold red]  @{user} — {result}")
            await asyncio.sleep(random.uniform(1.0, 2.5))

        # ── Comment ──
        if do_comment and media_id not in already_commented:
            comment = await generate_comment(
                caption, user, config,
                media_type=media_type,
                thumbnail_url=thumbnail_url,
                video_url=video_url,
            )
            if comment:
                result = await bridge.comment_post(media_id=media_id, text=comment)
                if "[SUCCESS]" in result:
                    stats["commented"] += 1
                    console.print(f"         [cyan]💬[/cyan] [italic]\"{comment}\"[/italic] — [dim]{url or media_id}[/dim]")
                elif "[SKIP]" in result:
                    pass
                else:
                    log.warning(f"         Comment failed: {result}")
                await asyncio.sleep(random.uniform(2.0, 4.0))

    return stats


async def main():
    username = os.environ.get("IG_USERNAME", "")
    password = os.environ.get("IG_PASSWORD", "")

    if not username or not password:
        username = input("Instagram username: ").strip()
        password = input("Instagram password: ").strip()

    # Parse flags
    once = "--once" in sys.argv
    no_comment = "--no-comment" in sys.argv
    interval = 300  # default 5 minutes
    for arg in sys.argv:
        if arg.startswith("--interval"):
            if "=" in arg:
                interval = int(arg.split("=")[1])
            else:
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    interval = int(sys.argv[idx + 1])

    # ── Beautiful startup ──
    banner = Text()
    banner.append("  AUTO-FEED BOT\n", style="bold bright_magenta")
    banner.append("  Like + Comment + Vision", style="dim")

    console.print()
    console.print(Panel(
        Align.center(banner),
        title="[bold yellow]Instagram Auto-Feed[/bold yellow]",
        subtitle="[dim]Powered by DeepSeek + OpenRouter Vision[/dim]",
        border_style="bright_magenta",
        box=box.DOUBLE_EDGE,
        padding=(1, 2),
    ))
    console.print()

    # Config table
    table = Table(box=box.ROUNDED, border_style="dim magenta", show_header=False, padding=(0, 2))
    table.add_column("Key", style="bold white", width=14)
    table.add_column("Value", style="green")
    table.add_row("Account", f"@{username}")
    table.add_row("Mode", "One-shot" if once else f"Loop every {interval}s")
    table.add_row("Commenting", "[red]OFF[/red]" if no_comment else "[green]ON[/green] (AI + Vision)")
    table.add_row("Vision", f"[green]{len(VISION_MODELS)} free models[/green] (round-robin)")
    table.add_row("Primary", f"[dim]{VISION_MODELS[0].split('/')[-1]}[/dim]")
    console.print(Align.center(table))
    console.print()

    # Login
    with console.status("[bold cyan]Logging in to Instagram...", spinner="dots12"):
        bridge = InstagrapiBridge(username, password)
    config = SwarmConfig.from_env()
    console.print("[green]✓ Logged in successfully[/green]")
    console.print()

    loop_count = 0
    total_stats = {"liked": 0, "skipped": 0, "commented": 0, "failed": 0}

    while True:
        loop_count += 1
        console.print(Rule(f"[bold bright_magenta]Pass #{loop_count}[/bold bright_magenta]", style="bright_magenta"))
        console.print()

        try:
            stats = await process_feed(bridge, config, do_comment=not no_comment)
            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

            # Stats table
            st = Table(
                title=f"[bold]Pass #{loop_count}[/bold]",
                box=box.ROUNDED, border_style="bright_green", show_lines=True,
            )
            st.add_column("Metric", style="bold", width=12)
            st.add_column("Pass", justify="center", style="cyan", width=8)
            st.add_column("Total", justify="center", style="bold green", width=8)
            st.add_row("Liked", str(stats["liked"]), str(total_stats["liked"]))
            st.add_row("Commented", str(stats["commented"]), str(total_stats["commented"]))
            st.add_row("Skipped", str(stats["skipped"]), str(total_stats["skipped"]))
            st.add_row("Failed", str(stats["failed"]), str(total_stats["failed"]))
            console.print()
            console.print(Align.center(st))
            console.print()
        except Exception as e:
            console.print(f"[bold red]Error in pass #{loop_count}: {e}[/bold red]")

        if once:
            break

        # Countdown
        with Progress(
            SpinnerColumn("dots"),
            TextColumn("[bold magenta]Next scan in"),
            BarColumn(bar_width=30, style="magenta", complete_style="bright_magenta"),
            TaskProgressColumn(),
            TextColumn("[dim]seconds[/dim]"),
            console=console,
        ) as progress:
            task = progress.add_task("countdown", total=interval)
            for _ in range(interval):
                await asyncio.sleep(1)
                progress.update(task, advance=1)
        console.print()

    console.print(Rule("[bold green]Session Complete[/bold green]", style="green"))
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
