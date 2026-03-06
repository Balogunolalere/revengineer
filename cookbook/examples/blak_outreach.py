"""BlakStudio.dev — Intelligent Instagram DM Outreach Bot

Continuously scans your feed. For each poster, it:
1. Fetches their full business profile
2. Uses AI to qualify them (are they a business that needs web/email/branding?)
3. Generates a hyper-personalized DM pitching the exact BlakStudio service they need
4. Sends the DM (dedup tracked — never DMs the same person twice)
5. Also likes the post + leaves a smart comment (reuses auto_like logic)

Usage:
    export IG_USERNAME="you@example.com"
    export IG_PASSWORD="your-password"
    export OPENROUTER_API_KEY="sk-or-v1-..."   # free key from openrouter.ai

    # Run once (scan feed, qualify, DM, exit)
    python -m cookbook.examples.blak_outreach --once

    # Run continuously (every 10 min by default)
    python -m cookbook.examples.blak_outreach

    # Custom interval + no commenting (DM only)
    python -m cookbook.examples.blak_outreach --interval 900 --no-comment

    # DM dry-run (qualify + generate DM but don't send)
    python -m cookbook.examples.blak_outreach --dry-run
"""

import asyncio
import json
import os
import sys
import random
import logging
import re

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

If you cannot write a genuine personalized message, respond with SKIP.

Respond with ONLY the DM text, nothing else."""


COMMENT_SYSTEM = """You are a friendly Instagram user. Given a post's details, write a SHORT, natural comment (1-2 sentences max).

Rules:
- Be genuine and relevant to the post content
- No emojis spam — 0-2 emojis max
- No generic comments — reference something specific
- Match the vibe: professional for business posts, casual for personal
- Never be promotional or salesy (the DM handles that separately)
- If the post is clearly an ad/spam, respond with SKIP
- Keep it under 150 characters
- IMPORTANT: If the post is showcasing products they are selling (shoes, clothes, jewellery, bags, food, etc.), do NOT ask about price, size, cost, or availability. Instead, compliment the aesthetics, the color, the design, or how beautiful the item looks.
- Never ask "How much?", "What size?", "Where can I get this?" — just appreciate the beauty.

Respond with ONLY the comment text, nothing else."""


# ── LLM helpers ───────────────────────────────────────────────────

# ── Vision API config (OpenRouter free models) ───────────────────
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

VISION_DESCRIBE_PROMPT = """Describe this Instagram post image in 1-2 sentences. Focus on:
- What's shown (people, objects, scenery, food, fashion, etc.)
- The mood/vibe (professional, casual, celebratory, etc.)
- Any text overlays visible
Be concise and factual."""

async def llm_call(config: SwarmConfig, system: str, user_msg: str, max_tokens: int = 200, temp: float = 0.7) -> str | None:
    """Generic DeepSeek text completion call."""
    api_base = config.api_base or "http://localhost:8000/v1"
    try:
        async with httpx.AsyncClient(base_url=api_base, timeout=45) as client:
            resp = await client.post("/chat/completions", json={
                "model": config.default_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": temp,
                "max_tokens": max_tokens,
                "stream": False,
            })
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return None


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

    start = _vision_idx
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
                short_name = model.split('/')[-1]
                console.print(f"         [dim]👁️  Vision ({short_name}): {desc[:70]}...[/dim]")
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


async def verify_no_website(username: str, full_name: str, biography: str, external_url: str) -> dict:
    """Verify a business truly doesn't have a website by checking their links more carefully.

    Returns: {"has_website": bool, "url_found": str, "method": str}
    """
    # 1) If they have a real external_url, check if it's a legit domain (not linktree etc.)
    link_in_bio_services = [
        "linktr.ee", "linktree.com", "bio.link", "linkbio.co", "tap.bio",
        "campsite.bio", "solo.to", "carrd.co", "beacons.ai", "lnk.bio",
        "milkshake.app", "stan.store", "hoo.be", "snipfeed.co", "flowpage.com",
    ]

    if external_url:
        url_lower = external_url.lower()
        is_link_in_bio = any(svc in url_lower for svc in link_in_bio_services)
        if not is_link_in_bio:
            # They have a real website
            return {"has_website": True, "url_found": external_url, "method": "instagram_profile"}

    # 2) Check their bio for website mentions
    if biography:
        bio_lower = biography.lower()
        # Look for domain patterns in bio (e.g., "mybrand.com", "www.shop.ng")
        domain_pattern = r'(?:https?://)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?)'
        domains_in_bio = re.findall(domain_pattern, bio_lower)
        for domain in domains_in_bio:
            # Filter out social media domains — those aren't business websites
            social_domains = [
                "instagram.com", "facebook.com", "twitter.com", "tiktok.com",
                "youtube.com", "wa.me", "t.me", "snapchat.com", "whatsapp.com",
                "threads.net", "x.com",
            ]
            if not any(s in domain for s in social_domains + link_in_bio_services):
                return {"has_website": True, "url_found": domain, "method": "bio_text"}

    # 3) Quick Google-like check: try to fetch common domain patterns
    #    e.g., if username is "komfort.place", try "komfort.place" or "komfortplace.com"
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', username.lower())
    candidate_domains = [
        f"{clean_name}.com",
        f"{clean_name}.ng",
        f"{clean_name}.co",
        f"{clean_name}.store",
    ]
    # Also try the username with dots preserved
    if "." in username:
        candidate_domains.insert(0, username.lower())

    for domain in candidate_domains:
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
                resp = await client.head(f"https://{domain}")
                if resp.status_code < 400:
                    return {"has_website": True, "url_found": f"https://{domain}", "method": "domain_probe"}
        except Exception:
            pass
        try:
            async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
                resp = await client.head(f"http://{domain}")
                if resp.status_code < 400:
                    return {"has_website": True, "url_found": f"http://{domain}", "method": "domain_probe"}
        except Exception:
            pass

    return {"has_website": False, "url_found": "", "method": "none_found"}


async def qualify_profile(profile: dict, config: SwarmConfig) -> dict | None:
    """Ask LLM if this profile is worth reaching out to. Returns qualification dict or None."""
    user_msg = json.dumps(profile, indent=2)
    raw = await llm_call(config, QUALIFY_SYSTEM, user_msg, max_tokens=300, temp=0.3)
    if not raw:
        return None

    # Parse JSON from response (handle markdown fences)
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text)
        return result
    except json.JSONDecodeError:
        log.warning(f"Failed to parse qualification: {raw[:100]}")
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
) -> str | None:
    """Generate a comment with optional image analysis via vision API."""
    # Try vision analysis
    image_desc = None
    if thumbnail_url:
        image_desc = await describe_image(thumbnail_url)

    # Build context
    parts = [f"Username: @{username}"]
    if media_type:
        parts.append(f"Media type: {media_type}")
    if caption and len(caption.strip()) >= 5:
        parts.append(f"Caption: {caption[:500]}")
    if image_desc:
        parts.append(f"Image description: {image_desc}")

    if not caption and not image_desc:
        return None
    if not image_desc and (not caption or len(caption.strip()) < 10):
        return None

    user_msg = "\n".join(parts)
    comment = await llm_call(config, COMMENT_SYSTEM, user_msg, max_tokens=100, temp=0.8)
    if not comment or comment.upper() == "SKIP" or len(comment) < 3:
        return None
    if comment.startswith('"') and comment.endswith('"'):
        comment = comment[1:-1]
    return comment[:300]


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
        subtitle="[dim]Powered by DeepSeek + OpenRouter Vision[/dim]",
        border_style="bright_cyan",
        box=box.DOUBLE_EDGE,
        padding=(1, 2),
    )


def make_config_table(username: str, once: bool, interval: int, no_comment: bool, dry_run: bool):
    """Create config display table."""
    table = Table(box=box.ROUNDED, border_style="dim cyan", show_header=False, padding=(0, 2))
    table.add_column("Key", style="bold white", width=15)
    table.add_column("Value", style="green")
    table.add_row("Account", f"@{username}")
    table.add_row("Mode", "One-shot" if once else f"Loop every {interval}s")
    table.add_row("Commenting", "[red]OFF[/red]" if no_comment else "[green]ON[/green] (AI + Vision)")
    table.add_row("DM Mode", "[yellow]DRY RUN[/yellow]" if dry_run else "[green]LIVE[/green]")
    table.add_row("Vision", f"[green]{len(VISION_MODELS)} free models[/green] (round-robin)")
    table.add_row("Primary", f"[dim]{VISION_MODELS[0].split('/')[-1]}[/dim]")
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
        ("Qualified", "qualified"), ("DMs Sent", "dm_sent"),
        ("DM Skipped", "dm_skipped"), ("Skipped", "skipped"), ("Failed", "failed"),
    ]
    for label, key in rows:
        table.add_row(label, str(stats.get(key, 0)), str(total.get(key, 0)))
    return table


# ── Main outreach loop ────────────────────────────────────────────

async def process_feed(
    bridge: InstagrapiBridge,
    config: SwarmConfig,
    do_comment: bool = True,
    dry_run: bool = False,
) -> dict:
    """Single pass: fetch feed → like → vision → qualify → verify website → DM → comment."""
    stats = {
        "posts": 0, "liked": 0, "skipped": 0, "qualified": 0,
        "dm_sent": 0, "dm_skipped": 0, "commented": 0, "failed": 0,
    }

    # Load dedup sets
    liked_data = json.loads(await bridge.get_liked_posts())
    already_liked = set(str(i) for i in liked_data.get("liked_ids", []))

    commented_data = json.loads(await bridge.get_commented_posts())
    already_commented = set(str(i) for i in commented_data.get("commented_ids", []))

    dm_data = json.loads(await bridge.get_dm_sent())
    already_dmed = set(str(i) for i in dm_data.get("dm_sent_ids", []))

    processed_users = set()

    # Fetch feed with spinner
    with console.status("[bold cyan]Fetching timeline feed...", spinner="dots"):
        feed_raw = await bridge.get_timeline_feed(amount="30")
    feed_data = json.loads(feed_raw)
    posts = feed_data.get("posts", [])

    if not posts:
        console.print("[yellow]No posts found on timeline.[/yellow]")
        return stats

    stats["posts"] = len(posts)
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

        if not media_id or not user:
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
            await asyncio.sleep(random.uniform(1.0, 2.0))

        # ── Comment (with vision) ──
        if do_comment and media_id not in already_commented:
            comment = await generate_comment(
                caption, user, config,
                media_type=media_type, thumbnail_url=thumbnail_url,
            )
            if comment:
                result = await bridge.comment_post(media_id=media_id, text=comment)
                if "[SUCCESS]" in result:
                    stats["commented"] += 1
                    console.print(f"         [cyan]💬[/cyan] [italic]\"{comment}\"[/italic] — [dim]{url}[/dim]")
                await asyncio.sleep(random.uniform(2.0, 3.5))

        # ── Qualify + DM ──
        if user in processed_users:
            continue
        processed_users.add(user)

        # Fetch business profile
        with console.status(f"[dim]Checking @{user}...[/dim]", spinner="point"):
            profile_raw = await bridge.get_business_profile(username=user)
        if "[ERROR]" in profile_raw:
            continue

        profile = json.loads(profile_raw)
        user_id = profile.get("user_id", "")

        if user_id in already_dmed:
            console.print(f"         [dim]📩 Already DMed @{user}[/dim]")
            stats["dm_skipped"] += 1
            continue

        # AI qualification
        qual = await qualify_profile(profile, config)
        if not qual:
            continue

        qualify_decision = qual.get("qualify", "NO").upper()
        reason = qual.get("reason", "")
        services = qual.get("services_needed", [])
        priority = qual.get("priority", "low")

        if qualify_decision != "YES":
            console.print(f"         [dim red]✖ NOT QUALIFIED @{user}: {reason}[/dim red]")
            continue

        stats["qualified"] += 1

        # ── Website Verification ──
        # Before DMing about websites, verify they truly don't have one
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
            # Remove "website" from needed services if they have one
            if "website" in services:
                services.remove("website")
            # If only website was needed, skip entirely
            if not services:
                console.print(f"         [dim]Skipping — already has website, no other services needed[/dim]")
                continue
            # Re-adjust qualification reason
            qual["services_needed"] = services
        else:
            console.print(f"         [green]✓ No website confirmed[/green] for @{user}")

        priority_colors = {"high": "bold red", "medium": "yellow", "low": "dim"}
        pcolor = priority_colors.get(priority, "white")
        console.print(
            f"         [green]✅ QUALIFIED[/green] [bold]@{user}[/bold] "
            f"[{pcolor}][{priority.upper()}][/{pcolor}] — needs: [cyan]{', '.join(services)}[/cyan]"
        )
        console.print(f"         [dim]{reason}[/dim]")

        # Generate personalized DM
        dm_text = await generate_dm(profile, qual, config)
        if not dm_text:
            console.print(f"         [dim]⏭️  Skipped DM (generator returned skip)[/dim]")
            continue

        if dry_run:
            console.print(Panel(
                dm_text,
                title=f"[yellow]DRY RUN DM → @{user}[/yellow]",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(0, 1),
            ))
            stats["dm_sent"] += 1
            continue

        # Send the DM
        result = await bridge.send_dm(user_id=user_id, text=dm_text)
        if "[SUCCESS]" in result:
            stats["dm_sent"] += 1
            console.print(Panel(
                dm_text,
                title=f"[green]📩 DM SENT → @{user}[/green]",
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 1),
            ))
        elif "[SKIP]" in result:
            stats["dm_skipped"] += 1
            console.print(f"         [dim]📩 Already DMed @{user}[/dim]")
        else:
            stats["failed"] += 1
            console.print(f"         [bold red]❌ DM FAILED @{user}: {result}[/bold red]")

        await asyncio.sleep(random.uniform(3.0, 6.0))

    return stats


# ── Entrypoint ────────────────────────────────────────────────────

async def main():
    username = os.environ.get("IG_USERNAME", "")
    password = os.environ.get("IG_PASSWORD", "")

    if not username or not password:
        username = input("Instagram username: ").strip()
        password = input("Instagram password: ").strip()

    once = "--once" in sys.argv
    no_comment = "--no-comment" in sys.argv
    dry_run = "--dry-run" in sys.argv
    interval = 600  # 10 min default

    for arg in sys.argv:
        if arg.startswith("--interval"):
            if "=" in arg:
                interval = int(arg.split("=")[1])
            else:
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    interval = int(sys.argv[idx + 1])

    # ── Beautiful startup ──
    console.print()
    console.print(make_banner())
    console.print()
    console.print(Align.center(make_config_table(username, once, interval, no_comment, dry_run)))
    console.print()

    # Login with spinner
    with console.status("[bold cyan]Logging in to Instagram...", spinner="dots12"):
        bridge = InstagrapiBridge(username, password)
    config = SwarmConfig.from_env()
    console.print("[green]✓ Logged in successfully[/green]")
    console.print()

    loop_count = 0
    total = {
        "posts": 0, "liked": 0, "skipped": 0, "qualified": 0,
        "dm_sent": 0, "dm_skipped": 0, "commented": 0, "failed": 0,
    }

    while True:
        loop_count += 1
        console.print(Rule(f"[bold bright_cyan]Pass #{loop_count}[/bold bright_cyan]", style="bright_cyan"))
        console.print()

        try:
            stats = await process_feed(bridge, config, do_comment=not no_comment, dry_run=dry_run)
            for k in total:
                total[k] += stats.get(k, 0)

            console.print()
            console.print(Align.center(make_stats_table(stats, total, loop_count)))
            console.print()
        except Exception as e:
            console.print(f"[bold red]Error in pass #{loop_count}: {e}[/bold red]")

        if once:
            break

        # Countdown timer
        console.print()
        with Progress(
            SpinnerColumn("dots"),
            TextColumn("[bold blue]Next scan in"),
            BarColumn(bar_width=30, style="cyan", complete_style="bright_cyan"),
            TaskProgressColumn(),
            TextColumn("[dim]seconds[/dim]"),
            console=console,
        ) as progress:
            task = progress.add_task("countdown", total=interval)
            for tick in range(interval):
                await asyncio.sleep(1)
                progress.update(task, advance=1)
        console.print()

    # Final summary
    console.print(Rule("[bold green]Session Complete[/bold green]", style="green"))
    console.print(Align.center(make_stats_table(total, total, loop_count)))
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
