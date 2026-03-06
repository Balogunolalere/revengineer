"""
Campaign ledger — persistent SQLite-backed history of all outreach actions.

Prevents duplicate outreach across runs, enables follow-up campaigns,
and provides analytics for campaign reporting.

Schema:
    contacts     – every user we've interacted with
    actions      – every individual action (follow, like, DM)
    campaigns    – one row per swarm campaign run

Usage:
    ledger = CampaignLedger()            # creates ~/.revengineer/campaign.db
    ledger = CampaignLedger("my.db")     # custom path

    # Check before acting
    if ledger.was_contacted("12345"):
        skip(...)

    # Log actions
    ledger.log_action(user_id="12345", username="acme", action="dm", detail="Hey!")

    # Follow-up query
    leads = ledger.get_followup_candidates(days_since=3)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default ledger location
DEFAULT_DB = os.path.expanduser("~/.revengineer/campaign.db")


@dataclass
class ContactRecord:
    """A single contact with a user."""

    user_id: str
    username: str
    first_seen: float      # epoch timestamp
    last_action: float     # epoch timestamp
    actions: list[str]     # e.g. ["follow", "like", "dm"]
    dm_count: int
    follow_count: int
    like_count: int
    replied: bool
    notes: str


@dataclass
class LeadRecord:
    """A discovered lead with profile data and optional draft DM."""

    user_id: str
    username: str
    full_name: str
    bio: str
    follower_count: int
    following_count: int
    is_private: bool
    source: str            # how the lead was found (hashtag, search, explore)
    source_detail: str     # e.g. the hashtag name or search query
    score: float           # agent-assigned relevance score 0-1
    draft_dm: str          # pre-written DM text
    status: str            # new, queued, sent, failed, skipped
    campaign_id: str
    created_at: float
    updated_at: float
    notes: str


@dataclass
class QueuedDM:
    """A DM waiting to be sent."""

    id: int
    user_id: str
    username: str
    text: str
    status: str            # pending, sent, failed, cancelled
    attempts: int
    last_attempt: float | None
    created_at: float
    campaign_id: str


class CampaignLedger:
    """SQLite-backed persistent ledger for Instagram marketing campaigns.

    Tracks every user interaction across runs to prevent duplicate outreach,
    enable follow-up campaigns, and provide analytics.
    """

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS contacts (
                user_id     TEXT PRIMARY KEY,
                username    TEXT NOT NULL,
                first_seen  REAL NOT NULL,       -- epoch
                last_action REAL NOT NULL,        -- epoch
                dm_count    INTEGER DEFAULT 0,
                follow_count INTEGER DEFAULT 0,
                like_count  INTEGER DEFAULT 0,
                replied     INTEGER DEFAULT 0,    -- boolean (0/1)
                notes       TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS actions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                username    TEXT NOT NULL DEFAULT '',
                action      TEXT NOT NULL,         -- 'dm', 'follow', 'like', 'comment'
                detail      TEXT DEFAULT '',        -- DM text, comment text, media_id, etc.
                result      TEXT DEFAULT 'ok',      -- 'ok', 'failed', 'skipped'
                campaign_id TEXT DEFAULT '',
                timestamp   REAL NOT NULL,          -- epoch
                FOREIGN KEY (user_id) REFERENCES contacts(user_id)
            );

            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_id TEXT PRIMARY KEY,
                started_at  REAL NOT NULL,
                finished_at REAL,
                mode        TEXT DEFAULT 'live',    -- 'live' or 'dry_run'
                hashtags    TEXT DEFAULT '[]',       -- JSON list
                leads_found INTEGER DEFAULT 0,
                dms_sent    INTEGER DEFAULT 0,
                follows     INTEGER DEFAULT 0,
                likes       INTEGER DEFAULT 0,
                notes       TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_actions_user ON actions(user_id);
            CREATE INDEX IF NOT EXISTS idx_actions_ts ON actions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_actions_campaign ON actions(campaign_id);

            CREATE TABLE IF NOT EXISTS leads (
                user_id       TEXT PRIMARY KEY,
                username      TEXT NOT NULL DEFAULT '',
                full_name     TEXT DEFAULT '',
                bio           TEXT DEFAULT '',
                follower_count INTEGER DEFAULT 0,
                following_count INTEGER DEFAULT 0,
                is_private    INTEGER DEFAULT 0,
                source        TEXT DEFAULT '',
                source_detail TEXT DEFAULT '',
                score         REAL DEFAULT 0.0,
                draft_dm      TEXT DEFAULT '',
                status        TEXT DEFAULT 'new',
                campaign_id   TEXT DEFAULT '',
                created_at    REAL NOT NULL,
                updated_at    REAL NOT NULL,
                notes         TEXT DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status);
            CREATE INDEX IF NOT EXISTS idx_leads_score ON leads(score DESC);

            CREATE TABLE IF NOT EXISTS dm_queue (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT NOT NULL,
                username      TEXT NOT NULL DEFAULT '',
                text          TEXT NOT NULL,
                status        TEXT DEFAULT 'pending',
                attempts      INTEGER DEFAULT 0,
                last_attempt  REAL,
                created_at    REAL NOT NULL,
                campaign_id   TEXT DEFAULT '',
                FOREIGN KEY (user_id) REFERENCES contacts(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_dmq_status ON dm_queue(status);
        """)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── Write methods ─────────────────────────────────────────────

    def log_action(
        self,
        user_id: str,
        action: str,
        username: str = "",
        detail: str = "",
        result: str = "ok",
        campaign_id: str = "",
    ) -> None:
        """Record an action (dm, follow, like, comment) for a user.

        Automatically creates or updates the contact record.
        """
        now = time.time()
        cur = self._conn.cursor()

        # Upsert contact
        cur.execute(
            """
            INSERT INTO contacts (user_id, username, first_seen, last_action)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                last_action = excluded.last_action,
                username = CASE WHEN excluded.username != '' THEN excluded.username
                                ELSE contacts.username END
            """,
            (user_id, username, now, now),
        )

        # Increment the relevant counter
        col_map = {"dm": "dm_count", "follow": "follow_count", "like": "like_count"}
        col = col_map.get(action)
        if col and result == "ok":
            cur.execute(
                f"UPDATE contacts SET {col} = {col} + 1 WHERE user_id = ?",
                (user_id,),
            )

        # Insert action row
        cur.execute(
            """
            INSERT INTO actions (user_id, username, action, detail, result, campaign_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, username, action, detail[:2000], result, campaign_id, now),
        )
        self._conn.commit()

    def start_campaign(
        self,
        campaign_id: str,
        mode: str = "live",
        hashtags: list[str] | None = None,
    ) -> None:
        """Register the start of a new campaign run."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO campaigns (campaign_id, started_at, mode, hashtags)
            VALUES (?, ?, ?, ?)
            """,
            (campaign_id, time.time(), mode, json.dumps(hashtags or [])),
        )
        self._conn.commit()

    def finish_campaign(self, campaign_id: str, notes: str = "") -> None:
        """Mark a campaign as finished and compute final stats."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT action, result FROM actions WHERE campaign_id = ?",
            (campaign_id,),
        )
        rows = cur.fetchall()
        dms = sum(1 for r in rows if r["action"] == "dm" and r["result"] == "ok")
        follows = sum(1 for r in rows if r["action"] == "follow" and r["result"] == "ok")
        likes = sum(1 for r in rows if r["action"] == "like" and r["result"] == "ok")
        leads = len(set(r["user_id"] for r in self._conn.execute(
            "SELECT DISTINCT user_id FROM actions WHERE campaign_id = ?",
            (campaign_id,),
        )))

        self._conn.execute(
            """
            UPDATE campaigns SET finished_at = ?, leads_found = ?, dms_sent = ?,
                follows = ?, likes = ?, notes = ?
            WHERE campaign_id = ?
            """,
            (time.time(), leads, dms, follows, likes, notes, campaign_id),
        )
        self._conn.commit()

    def mark_replied(self, user_id: str) -> None:
        """Mark that a user has replied to our outreach."""
        self._conn.execute(
            "UPDATE contacts SET replied = 1 WHERE user_id = ?",
            (user_id,),
        )
        self._conn.commit()

    # ── Read / query methods ──────────────────────────────────────

    def was_contacted(self, user_id: str) -> bool:
        """Check if we've already sent a DM to this user."""
        row = self._conn.execute(
            "SELECT dm_count FROM contacts WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return bool(row and row["dm_count"] > 0)

    def was_engaged(self, user_id: str) -> bool:
        """Check if we've done ANY interaction with this user (follow, like, DM)."""
        return self._conn.execute(
            "SELECT 1 FROM contacts WHERE user_id = ?",
            (user_id,),
        ).fetchone() is not None

    def get_contact(self, user_id: str) -> ContactRecord | None:
        """Get full contact record for a user."""
        row = self._conn.execute(
            "SELECT * FROM contacts WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return None

        actions = [
            r["action"] for r in self._conn.execute(
                "SELECT action FROM actions WHERE user_id = ? ORDER BY timestamp",
                (user_id,),
            )
        ]
        return ContactRecord(
            user_id=row["user_id"],
            username=row["username"],
            first_seen=row["first_seen"],
            last_action=row["last_action"],
            actions=actions,
            dm_count=row["dm_count"],
            follow_count=row["follow_count"],
            like_count=row["like_count"],
            replied=bool(row["replied"]),
            notes=row["notes"],
        )

    def get_followup_candidates(
        self,
        days_since: int = 3,
        max_dms: int = 1,
    ) -> list[ContactRecord]:
        """Get users who were DMed but haven't replied and enough time has passed.

        Args:
            days_since: Minimum days since last DM before follow-up.
            max_dms: Only return users who've received at most this many DMs
                     (prevents harassment).
        """
        cutoff = time.time() - (days_since * 86400)
        rows = self._conn.execute(
            """
            SELECT * FROM contacts
            WHERE dm_count > 0
              AND dm_count <= ?
              AND replied = 0
              AND last_action < ?
            ORDER BY last_action ASC
            """,
            (max_dms, cutoff),
        ).fetchall()

        results = []
        for row in rows:
            actions = [
                r["action"] for r in self._conn.execute(
                    "SELECT action FROM actions WHERE user_id = ? ORDER BY timestamp",
                    (row["user_id"],),
                )
            ]
            results.append(ContactRecord(
                user_id=row["user_id"],
                username=row["username"],
                first_seen=row["first_seen"],
                last_action=row["last_action"],
                actions=actions,
                dm_count=row["dm_count"],
                follow_count=row["follow_count"],
                like_count=row["like_count"],
                replied=bool(row["replied"]),
                notes=row["notes"],
            ))
        return results

    def get_all_contacted_ids(self) -> set[str]:
        """Return set of all user_ids that have been DMed."""
        return {
            row["user_id"] for row in self._conn.execute(
                "SELECT user_id FROM contacts WHERE dm_count > 0"
            )
        }

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate stats across all campaigns."""
        total_contacts = self._conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        total_dms = self._conn.execute(
            "SELECT COALESCE(SUM(dm_count), 0) FROM contacts"
        ).fetchone()[0]
        total_follows = self._conn.execute(
            "SELECT COALESCE(SUM(follow_count), 0) FROM contacts"
        ).fetchone()[0]
        total_likes = self._conn.execute(
            "SELECT COALESCE(SUM(like_count), 0) FROM contacts"
        ).fetchone()[0]
        total_replied = self._conn.execute(
            "SELECT COUNT(*) FROM contacts WHERE replied = 1"
        ).fetchone()[0]
        total_campaigns = self._conn.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0]

        reply_rate = (total_replied / total_dms * 100) if total_dms > 0 else 0

        return {
            "total_contacts": total_contacts,
            "total_dms": total_dms,
            "total_follows": total_follows,
            "total_likes": total_likes,
            "total_replied": total_replied,
            "reply_rate": f"{reply_rate:.1f}%",
            "total_campaigns": total_campaigns,
        }

    def get_history_summary(self) -> str:
        """Get a human-readable summary for agents to consume.

        Returned as a formatted string agents can read in their context.
        """
        stats = self.get_stats()
        contacted = self.get_all_contacted_ids()
        followups = self.get_followup_candidates(days_since=3)

        lines = [
            "## Campaign History",
            f"- Total campaigns run: {stats['total_campaigns']}",
            f"- Users contacted (DM): {stats['total_dms']}",
            f"- Users followed: {stats['total_follows']}",
            f"- Posts liked: {stats['total_likes']}",
            f"- Replies received: {stats['total_replied']} ({stats['reply_rate']})",
            f"- Users already messaged (DO NOT DM AGAIN): {len(contacted)}",
        ]

        if contacted:
            # Show the most recent 20 for agent awareness
            recent = self._conn.execute(
                """SELECT user_id, username FROM contacts
                   WHERE dm_count > 0 ORDER BY last_action DESC LIMIT 20"""
            ).fetchall()
            lines.append("\n### Recently Messaged (skip these):")
            for r in recent:
                lines.append(f"  - @{r['username']} (id: {r['user_id']})")

        if followups:
            lines.append(f"\n### Follow-Up Candidates ({len(followups)} users):")
            lines.append("These users were DMed 3+ days ago but haven't replied.")
            for c in followups[:15]:
                lines.append(
                    f"  - @{c.username} (id: {c.user_id}) — "
                    f"{c.dm_count} DM(s), last contact {_days_ago(c.last_action)} days ago"
                )

        return "\n".join(lines)


    # ── Lead management ────────────────────────────────────────

    def save_lead(
        self,
        user_id: str,
        username: str = "",
        full_name: str = "",
        bio: str = "",
        follower_count: int = 0,
        following_count: int = 0,
        is_private: bool = False,
        source: str = "",
        source_detail: str = "",
        score: float = 0.0,
        draft_dm: str = "",
        status: str = "new",
        campaign_id: str = "",
        notes: str = "",
    ) -> None:
        """Save or update a lead with profile data and optional draft DM."""
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO leads
                (user_id, username, full_name, bio, follower_count, following_count,
                 is_private, source, source_detail, score, draft_dm, status,
                 campaign_id, created_at, updated_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username = CASE WHEN excluded.username != '' THEN excluded.username
                                ELSE leads.username END,
                full_name = CASE WHEN excluded.full_name != '' THEN excluded.full_name
                                 ELSE leads.full_name END,
                bio = CASE WHEN excluded.bio != '' THEN excluded.bio ELSE leads.bio END,
                follower_count = CASE WHEN excluded.follower_count > 0
                                      THEN excluded.follower_count
                                      ELSE leads.follower_count END,
                following_count = CASE WHEN excluded.following_count > 0
                                       THEN excluded.following_count
                                       ELSE leads.following_count END,
                is_private = excluded.is_private,
                score = CASE WHEN excluded.score > 0 THEN excluded.score ELSE leads.score END,
                draft_dm = CASE WHEN excluded.draft_dm != '' THEN excluded.draft_dm
                                ELSE leads.draft_dm END,
                status = CASE WHEN leads.status IN ('sent') THEN leads.status
                              ELSE excluded.status END,
                updated_at = excluded.updated_at,
                notes = CASE WHEN excluded.notes != '' THEN excluded.notes ELSE leads.notes END
            """,
            (user_id, username, full_name, bio, follower_count, following_count,
             int(is_private), source, source_detail, score, draft_dm, status,
             campaign_id, now, now, notes),
        )
        self._conn.commit()

    def get_lead(self, user_id: str) -> LeadRecord | None:
        """Get a lead record by user ID."""
        row = self._conn.execute(
            "SELECT * FROM leads WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return None
        return LeadRecord(
            user_id=row["user_id"],
            username=row["username"],
            full_name=row["full_name"],
            bio=row["bio"],
            follower_count=row["follower_count"],
            following_count=row["following_count"],
            is_private=bool(row["is_private"]),
            source=row["source"],
            source_detail=row["source_detail"],
            score=row["score"],
            draft_dm=row["draft_dm"],
            status=row["status"],
            campaign_id=row["campaign_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            notes=row["notes"],
        )

    def get_leads(
        self,
        status: str | None = None,
        min_score: float = 0.0,
        limit: int = 50,
    ) -> list[LeadRecord]:
        """Get leads, optionally filtered by status and minimum score."""
        query = "SELECT * FROM leads WHERE score >= ?"
        params: list[Any] = [min_score]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY score DESC, created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [
            LeadRecord(
                user_id=r["user_id"],
                username=r["username"],
                full_name=r["full_name"],
                bio=r["bio"],
                follower_count=r["follower_count"],
                following_count=r["following_count"],
                is_private=bool(r["is_private"]),
                source=r["source"],
                source_detail=r["source_detail"],
                score=r["score"],
                draft_dm=r["draft_dm"],
                status=r["status"],
                campaign_id=r["campaign_id"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                notes=r["notes"],
            )
            for r in rows
        ]

    def get_lead_stats(self) -> dict[str, int]:
        """Get aggregate lead counts by status."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM leads GROUP BY status"
        ).fetchall()
        result = {r["status"]: r["cnt"] for r in rows}
        result["total"] = sum(result.values())
        return result

    # ── DM queue management ───────────────────────────────────────

    def queue_dm(
        self,
        user_id: str,
        text: str,
        username: str = "",
        campaign_id: str = "",
    ) -> int:
        """Queue a DM for later delivery. Returns the queue entry ID."""
        now = time.time()
        # Don't queue duplicates — if a pending DM already exists for this user, update it
        existing = self._conn.execute(
            "SELECT id FROM dm_queue WHERE user_id = ? AND status = 'pending'",
            (user_id,),
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE dm_queue SET text = ?, updated_at = ? WHERE id = ?",
                (text, now, existing["id"]),
            )
            self._conn.commit()
            return existing["id"]

        cur = self._conn.execute(
            """
            INSERT INTO dm_queue (user_id, username, text, status, created_at, campaign_id)
            VALUES (?, ?, ?, 'pending', ?, ?)
            """,
            (user_id, username, text, now, campaign_id),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_pending_dms(self, limit: int = 20) -> list[QueuedDM]:
        """Get pending DMs in queue, oldest first."""
        rows = self._conn.execute(
            """
            SELECT * FROM dm_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            QueuedDM(
                id=r["id"],
                user_id=r["user_id"],
                username=r["username"],
                text=r["text"],
                status=r["status"],
                attempts=r["attempts"],
                last_attempt=r["last_attempt"],
                created_at=r["created_at"],
                campaign_id=r["campaign_id"],
            )
            for r in rows
        ]

    def mark_dm_sent(self, queue_id: int) -> None:
        """Mark a queued DM as successfully sent."""
        self._conn.execute(
            "UPDATE dm_queue SET status = 'sent', last_attempt = ? WHERE id = ?",
            (time.time(), queue_id),
        )
        self._conn.commit()

    def mark_dm_failed(self, queue_id: int) -> None:
        """Increment attempt count for a failed DM. After 3 attempts, mark as 'failed'."""
        self._conn.execute(
            """
            UPDATE dm_queue SET
                attempts = attempts + 1,
                last_attempt = ?,
                status = CASE WHEN attempts + 1 >= 3 THEN 'failed' ELSE 'pending' END
            WHERE id = ?
            """,
            (time.time(), queue_id),
        )
        self._conn.commit()

    def get_dm_queue_stats(self) -> dict[str, int]:
        """Get DM queue counts by status."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM dm_queue GROUP BY status"
        ).fetchall()
        result = {r["status"]: r["cnt"] for r in rows}
        result["total"] = sum(result.values())
        return result


def _days_ago(epoch: float) -> int:
    """How many days ago was this timestamp."""
    return max(0, int((time.time() - epoch) / 86400))
