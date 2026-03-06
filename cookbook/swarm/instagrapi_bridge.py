import asyncio
import re
import json
import logging
import os
import pathlib
from typing import Any
from instagrapi import Client

from .tool_registry import ToolDef
from .campaign_ledger import CampaignLedger

logger = logging.getLogger("instagrapi_bridge")

def _to_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)

class InstagrapiBridge:
    def __init__(self, username: str, password: str, settings_file: str = "", dm_enabled: bool = True):
        self.username = username
        self.password = password
        # Per-account session file to avoid cross-account session leaks
        if settings_file:
            self.settings_file = settings_file
        else:
            safe = username.replace("@", "_").replace(".", "_").replace(" ", "_")
            self.settings_file = f"ig_settings_{safe}.json"
        self.dm_enabled = dm_enabled
        self.cl = Client()
        self.ledger = CampaignLedger()
        self.campaign_id = "test-campaign" # Usually set dynamically
        self.login()

    def login(self):
        try:
            if os.path.exists(self.settings_file):
                self.cl.load_settings(self.settings_file)
                # Verify the cached session belongs to this username
                old = self.cl.username
                if old and old != self.username:
                    logger.warning(f"Session file was for @{old}, re-authenticating as @{self.username}")
                    self.cl = Client()  # fresh client, discard stale session
            
            # This login will use the cached session if valid, or login if not
            self.cl.login(self.username, self.password)
            self.cl.dump_settings(self.settings_file)
            logger.info(f"Instagrapi logged in as @{self.username} (session: {self.settings_file})")
        except Exception as e:
            logger.error(f"Instagrapi login failed: {e}")
            raise
            
    async def _run_async(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # --- Tool Endpoints ---

    async def get_profile(self, username: str = "", **kwargs) -> str:
        """Fetch a specific user's profile."""
        if not username:
            return "[ERROR] missing username"
        try:
            user = await self._run_async(self.cl.user_info_by_username, username)
            return _to_json(getattr(user, "model_dump", user.dict)())
        except Exception as e:
            return f"[ERROR] Failed to fetch profile: {e}"

    async def hashtag_feed(self, hashtag: str = "", amount: str = "27", **kwargs) -> str:
        """Fetch recent posts from a hashtag."""
        if not hashtag:
            return "[ERROR] missing hashtag"
        hashtag = hashtag.replace("#", "")
        amount_int = int(amount) if str(amount).isdigit() else 27
        try:
            medias = await self._run_async(self.cl.hashtag_medias_top, hashtag, amount_int)
            results = []
            for m in medias:
                results.append({
                    "id": m.pk,
                    "code": m.code,
                    "caption_text": m.caption_text,
                    "user": getattr(m.user, "model_dump", m.user.dict)() if m.user else None,
                    "like_count": m.like_count,
                    "comment_count": m.comment_count
                })
            return _to_json(results)
        except Exception as e:
            return f"[ERROR] Failed to fetch hashtag feed: {e}"

    async def search_users(self, query: str = "", amount: str = "20", **kwargs) -> str:
        """Search users generically."""
        if not query:
            return "[ERROR] missing query"
        amount_int = int(amount) if str(amount).isdigit() else 20
        try:
            users = await self._run_async(self.cl.search_users, query, amount=amount_int)
            return _to_json([getattr(u, "model_dump", u.dict)() for u in users])
        except Exception as e:
            return f"[ERROR] Failed to search users: {e}"

    async def save_lead(self, user_id: str = "", username: str = "", full_name: str = "", bio: str = "", follower_count: str = "0", is_private: str = "false", source: str = "", score: str = "0.5", notes: str = "", **kwargs) -> str:
        if not user_id:
            return "[ERROR] user_id is required."
        
        # Anti-hallucination validation
        if "123456789" in str(user_id) or str(user_id) in ["123", "12345", "11111111"] or len(str(user_id)) < 4:
            return f"[ERROR] user_id '{user_id}' appears to be fake/hallucinated. Leads must have real IDs from actual execution results."
        if "lagos" in username.lower() and "owner" in username.lower():
            return f"[ERROR] username '{username}' appears to be hallucinated."

        try:
            follower_count_int = int(follower_count) if isinstance(follower_count, str) and follower_count.isdigit() else 0
        except ValueError:
            follower_count_int = 0
            
        try:
            is_priv = isinstance(is_private, str) and is_private.lower() in ("true", "1", "yes")
            self.ledger.save_lead(
                user_id=str(user_id),
                username=username,
                full_name=full_name,
                bio=bio[:1000],
                follower_count=follower_count_int,
                following_count=0,
                is_private=is_priv,
                source=source,
                source_detail="",
                score=float(score),
                draft_dm=notes[:2000], # Draft DM reusing notes param
                status="new",
                campaign_id=self.campaign_id,
                notes=notes,
            )
            return f"[SUCCESS] Lead {username} saved locally."
        except Exception as e:
            return f"[ERROR] {e}"

    async def get_timeline_feed(self, amount: str = "20", **kwargs) -> str:
        """Fetch posts from the authenticated user's timeline feed."""
        amount_int = int(amount) if str(amount).isdigit() else 20
        try:
            medias = await self._run_async(self.cl.get_timeline_feed)
            items = medias.get("feed_items", []) if isinstance(medias, dict) else []
            # Also try the direct media list approach
            if not items:
                try:
                    medias_list = await self._run_async(
                        self.cl.get_timeline_feed
                    )
                    if isinstance(medias_list, list):
                        items = medias_list[:amount_int]
                except Exception:
                    pass
            results = []
            for item in items[:amount_int]:
                media = item.get("media_or_ad", item) if isinstance(item, dict) else item
                if isinstance(media, dict):
                    code = media.get("code", "")
                    # Extract image/video URLs for vision analysis
                    image_versions = media.get("image_versions2", {}) or {}
                    candidates = image_versions.get("candidates", [])
                    thumbnail = candidates[0]["url"] if candidates else ""
                    video_versions = media.get("video_versions", []) or []
                    video_url = video_versions[0]["url"] if video_versions else ""
                    # media_type: 1=photo, 2=video, 8=carousel
                    media_type = media.get("media_type", 0)
                    results.append({
                        "id": str(media.get("pk", media.get("id", ""))),
                        "code": code,
                        "url": f"https://www.instagram.com/p/{code}/" if code else "",
                        "caption": (media.get("caption", {}) or {}).get("text", "")[:200] if isinstance(media.get("caption"), dict) else str(media.get("caption_text", ""))[:200],
                        "user": media.get("user", {}).get("username", "") if isinstance(media.get("user"), dict) else "",
                        "like_count": media.get("like_count", 0),
                        "has_liked": media.get("has_liked", False),
                        "media_type": "photo" if media_type == 1 else "video" if media_type == 2 else "carousel" if media_type == 8 else "unknown",
                        "thumbnail_url": thumbnail,
                        "video_url": video_url,
                    })
                else:
                    # It's a Media object from instagrapi
                    try:
                        code = getattr(media, "code", "")
                        thumb = str(media.thumbnail_url) if getattr(media, "thumbnail_url", None) else ""
                        vid = str(media.video_url) if getattr(media, "video_url", None) else ""
                        mtype = getattr(media, "media_type", 0)
                        results.append({
                            "id": str(media.pk),
                            "code": code,
                            "url": f"https://www.instagram.com/p/{code}/" if code else "",
                            "caption": (media.caption_text or "")[:200],
                            "user": media.user.username if media.user else "",
                            "like_count": getattr(media, "like_count", 0),
                            "has_liked": getattr(media, "has_liked", False),
                            "media_type": "photo" if mtype == 1 else "video" if mtype == 2 else "carousel" if mtype == 8 else "unknown",
                            "thumbnail_url": thumb,
                            "video_url": vid,
                        })
                    except Exception:
                        continue
            return _to_json({"count": len(results), "posts": results})
        except Exception as e:
            return f"[ERROR] Failed to fetch timeline feed: {e}"

    async def like_post(self, media_id: str = "", **kwargs) -> str:
        """Like a post by media ID. Tracks liked posts to prevent duplicates."""
        if not media_id:
            return "[ERROR] missing media_id"
        # Dedup check
        liked_ids = self._load_liked_ids()
        if media_id in liked_ids:
            return f"[SKIP] Post {media_id} already liked previously."
        try:
            # Try to get the shortcode for a URL
            post_url = ""
            try:
                info = await self._run_async(self.cl.media_info, media_id)
                if info and getattr(info, "code", None):
                    post_url = f" (https://www.instagram.com/p/{info.code}/)"
            except Exception:
                pass

            result = await self._run_async(self.cl.media_like, media_id)
            if result:
                self._save_liked_id(media_id)
                return f"[SUCCESS] Liked post {media_id}{post_url}."
            else:
                return f"[WARN] media_like returned False for {media_id}{post_url} (may already be liked)."
        except Exception as e:
            err = str(e)
            if "already liked" in err.lower() or "has_liked" in err.lower():
                self._save_liked_id(media_id)  # record it so we skip next time
                return f"[SKIP] Post {media_id} was already liked."
            return f"[ERROR] Failed to like post {media_id}: {e}"

    async def get_liked_posts(self, **kwargs) -> str:
        """Return list of all previously liked media IDs from the local tracker."""
        liked_ids = self._load_liked_ids()
        return _to_json({"count": len(liked_ids), "liked_ids": list(liked_ids)})

    # --- Liked-post dedup persistence ---

    _LIKED_FILE = "liked_posts.json"

    def _load_liked_ids(self) -> set:
        try:
            with open(self._LIKED_FILE) as f:
                data = json.load(f)
                return set(str(i) for i in data)
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    def _save_liked_id(self, media_id: str):
        liked = self._load_liked_ids()
        liked.add(str(media_id))
        with open(self._LIKED_FILE, "w") as f:
            json.dump(sorted(liked), f, indent=2)

    async def get_campaign_history(self, **kwargs) -> str:
        try:
            return self.ledger.get_history_summary()
        except Exception as e:
            return f"[ERROR] Failed to read ledger: {e}"

    # --- Commenting ---

    async def comment_post(self, media_id: str = "", text: str = "", **kwargs) -> str:
        """Comment on a post by media ID. Tracks commented posts to prevent duplicates."""
        if not media_id:
            return "[ERROR] missing media_id"
        if not text:
            return "[ERROR] missing comment text"

        # Dedup check
        commented = self._load_commented_ids()
        if media_id in commented:
            return f"[SKIP] Already commented on post {media_id}."

        try:
            post_url = ""
            try:
                info = await self._run_async(self.cl.media_info, media_id)
                if info and getattr(info, "code", None):
                    post_url = f" (https://www.instagram.com/p/{info.code}/)"
            except Exception:
                pass

            result = await self._run_async(self.cl.media_comment, media_id, text)
            if result:
                self._save_commented_id(media_id)
                return f"[SUCCESS] Commented on {media_id}{post_url}: \"{text[:80]}\""
            else:
                return f"[WARN] media_comment returned falsy for {media_id}"
        except Exception as e:
            return f"[ERROR] Failed to comment on {media_id}: {e}"

    async def get_commented_posts(self, **kwargs) -> str:
        """Return list of all previously commented media IDs from the local tracker."""
        commented = self._load_commented_ids()
        return _to_json({"count": len(commented), "commented_ids": list(commented)})

    _COMMENTED_FILE = "commented_posts.json"

    def _load_commented_ids(self) -> set:
        try:
            with open(self._COMMENTED_FILE) as f:
                data = json.load(f)
                return set(str(i) for i in data)
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    def _save_commented_id(self, media_id: str):
        commented = self._load_commented_ids()
        commented.add(str(media_id))
        with open(self._COMMENTED_FILE, "w") as f:
            json.dump(sorted(commented), f, indent=2)

    # --- DMs ---

    _DM_SENT_FILE = "dm_sent.json"

    def _load_dm_sent_ids(self) -> set:
        try:
            with open(self._DM_SENT_FILE) as f:
                data = json.load(f)
                return set(str(i) for i in data)
        except (FileNotFoundError, json.JSONDecodeError):
            return set()

    def _save_dm_sent_id(self, user_id: str):
        sent = self._load_dm_sent_ids()
        sent.add(str(user_id))
        with open(self._DM_SENT_FILE, "w") as f:
            json.dump(sorted(sent), f, indent=2)

    async def send_dm(self, user_id: str = "", text: str = "", **kwargs) -> str:
        """Send a direct message to a user by their numeric user ID. Dedup tracked."""
        if not user_id:
            return "[ERROR] missing user_id"
        if not text:
            return "[ERROR] missing message text"

        sent = self._load_dm_sent_ids()
        if user_id in sent:
            return f"[SKIP] Already sent DM to user {user_id}."

        try:
            result = await self._run_async(self.cl.direct_send, text, user_ids=[int(user_id)])
            if result:
                self._save_dm_sent_id(user_id)
                return f"[SUCCESS] DM sent to user {user_id}: \"{text[:80]}...\""
            else:
                return f"[WARN] direct_send returned falsy for user {user_id}"
        except Exception as e:
            return f"[ERROR] Failed to DM user {user_id}: {e}"

    async def get_dm_sent(self, **kwargs) -> str:
        """Return list of user IDs we've already DMed."""
        sent = self._load_dm_sent_ids()
        return _to_json({"count": len(sent), "dm_sent_ids": list(sent)})

    async def get_business_profile(self, username: str = "", **kwargs) -> str:
        """Fetch an enriched profile with business signals for outreach qualification."""
        if not username:
            return "[ERROR] missing username"
        try:
            user = await self._run_async(self.cl.user_info_by_username, username)
            data = getattr(user, "model_dump", user.dict)()

            # Extract the business-relevant fields into a clean summary
            profile = {
                "user_id": str(data.get("pk", "")),
                "username": data.get("username", ""),
                "full_name": data.get("full_name", ""),
                "biography": data.get("biography", ""),
                "follower_count": data.get("follower_count", 0),
                "following_count": data.get("following_count", 0),
                "media_count": data.get("media_count", 0),
                "is_business": data.get("is_business", False),
                "is_private": data.get("is_private", False),
                "is_verified": data.get("is_verified", False),
                "category_name": data.get("category_name", ""),
                "business_category_name": data.get("business_category_name", ""),
                "contact_phone_number": data.get("contact_phone_number", ""),
                "public_email": data.get("public_email", ""),
                "public_phone_number": data.get("public_phone_number", ""),
                "external_url": str(data.get("external_url", "") or ""),
                "bio_links": [str(l) for l in (data.get("bio_links", []) or [])],
            }

            # Compute business signals
            has_website = bool(profile["external_url"])
            has_email = bool(profile["public_email"])
            has_phone = bool(profile["public_phone_number"] or profile["contact_phone_number"])
            profile["signals"] = {
                "has_website": has_website,
                "has_business_email": has_email,
                "has_phone": has_phone,
                "is_business_account": profile["is_business"],
                "needs_website": profile["is_business"] and not has_website,
                "needs_email": profile["is_business"] and not has_email,
                "engagement_ratio": round(
                    (profile["follower_count"] / max(profile["media_count"], 1)), 1
                ),
            }

            return _to_json(profile)
        except Exception as e:
            return f"[ERROR] Failed to fetch business profile: {e}"


def get_instagrapi_tools(bridge: InstagrapiBridge) -> list[ToolDef]:
    return [
        ToolDef(
            name="ig_get_profile",
            description="Fetch a user's Instagram profile by username.",
            parameters={"username": "Exact username to lookup."},
            fn=bridge.get_profile,
        ),
        ToolDef(
            name="ig_hashtag_feed",
            description="Fetch recent/top posts from a specific hashtag.",
            parameters={"hashtag": "The hashtag string to search.", "amount": "Number of posts to return (default 20)"},
            fn=bridge.hashtag_feed,
        ),
        ToolDef(
            name="ig_search_users",
            description="Search Instagram users generically by query.",
            parameters={"query": "The search query, e.g. 'founder london'.", "amount": "Number of users (default 20)"},
            fn=bridge.search_users,
        ),
        ToolDef(
            name="ig_timeline_feed",
            description="Fetch posts from the logged-in user's home timeline feed. Returns a list of posts with id, code, caption, user, like_count, and has_liked fields.",
            parameters={"amount": "Max number of posts to return (default 20)"},
            fn=bridge.get_timeline_feed,
        ),
        ToolDef(
            name="ig_like_post",
            description="Like a single post by its media ID. Automatically skips posts that have already been liked (tracked in liked_posts.json).",
            parameters={"media_id": "The numeric media/post ID to like (required)"},
            fn=bridge.like_post,
        ),
        ToolDef(
            name="ig_get_liked_posts",
            description="Get the list of all media IDs that have been liked previously (from local tracker). Use this to avoid liking duplicates.",
            parameters={},
            fn=bridge.get_liked_posts,
        ),
        ToolDef(
            name="ig_save_lead",
            description="Save a promising profile to the local database.",
            parameters={
                "user_id": "Numeric user ID (required)",
                "username": "Username",
                "full_name": "Full name",
                "bio": "Bio snippet",
                "follower_count": "Number of followers",
                "is_private": "'true' or 'false'",
                "source": "Where you found them",
                "score": "Score (0.0 to 1.0)",
                "notes": "Draft DM or reason for saving"
            },
            fn=bridge.save_lead,
        ),
        ToolDef(
            name="ig_campaign_history",
            description="Get a summary of past contacted users.",
            parameters={},
            fn=bridge.get_campaign_history,
        ),
        ToolDef(
            name="ig_comment_post",
            description="Comment on a single post by its media ID. Automatically skips posts already commented on (tracked in commented_posts.json).",
            parameters={"media_id": "The numeric media/post ID to comment on (required)", "text": "The comment text to post (required)"},
            fn=bridge.comment_post,
        ),
        ToolDef(
            name="ig_get_commented_posts",
            description="Get the list of all media IDs that have been commented on previously (from local tracker).",
            parameters={},
            fn=bridge.get_commented_posts,
        ),
        ToolDef(
            name="ig_send_dm",
            description="Send a direct message to a user by their numeric user ID. Automatically skips users already DMed (tracked in dm_sent.json).",
            parameters={"user_id": "Numeric user ID (required)", "text": "The DM text to send (required)"},
            fn=bridge.send_dm,
        ),
        ToolDef(
            name="ig_get_dm_sent",
            description="Get the list of user IDs that have already been sent a DM (from local tracker).",
            parameters={},
            fn=bridge.get_dm_sent,
        ),
        ToolDef(
            name="ig_get_business_profile",
            description="Fetch an enriched profile with business signals: is_business, has_website, has_email, needs_website, needs_email, engagement_ratio, category, etc.",
            parameters={"username": "Exact username to lookup (required)"},
            fn=bridge.get_business_profile,
        ),
    ]
