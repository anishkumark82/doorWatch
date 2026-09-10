# notify.py
import time
from collections import deque
import subprocess
import os
from config import PIPER_MODEL, NTFY_TOPIC, HA_URL, HA_TOKEN, ALEXA_NOTIFY_ENTITIES
import uuid
import requests

def send_push_notification(text):
    """Send a push notification via ntfy.sh."""
    requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", data=text.encode("utf-8"))

def build_announcement(name=None, category=None, label=None):
    """Produce the human-facing text for a detection outcome.

    name     -- set when ArcFace matched a known enrolled person
    category -- set when unknown: either "possible_delivery" or "regular_visitor"
    label    -- the specific CLIP guess (e.g. "amazon_delivery"), kept for
                logging only -- not used to decide the wording here, since
                brand-level accuracy isn't reliable enough to announce on directly
    """
    if name:
        return f"{name.capitalize()} is here"
    if category == "possible_delivery":
        return "Possible delivery at the door"
    return "Unknown visitor at the door"

def speak_announcement(text):
    url = f"{HA_URL}/api/services/notify/send_message"
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    for entity_id in ALEXA_NOTIFY_ENTITIES:
        payload = {"entity_id": entity_id, "message": text}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

class AnnouncementGate:
    """Decides whether a detection outcome should actually be announced.

    Two problems this solves:
      1. Flip-flopping: borderline similarity scores (or a person turning
         slightly off-angle mid-approach) can make the same real person
         alternate between "known" and "unknown" from one check to the next.
         A strict "must repeat N times in a row" rule would never fire in
         that case, since the count resets on every flip. Instead, this uses
         a short rolling window and requires an outcome to appear at least
         min_hits times *within* that window -- tolerant of occasional
         misses, still resistant to one-off noise.
      2. Repeat-announcing: without a cooldown, a person standing at the
         door for 30 seconds would trigger a fresh announcement on every
         single check (once per second, per CHECK_INTERVAL).

    Multiple people at once are handled by keying all state on the outcome
    itself (e.g. ("known", "anish") vs ("unknown", "possible_delivery")),
    not on detection order or box index -- so two different people never
    interfere with each other's tracking, and a person's identity/category
    stays correctly tracked even if their position in the frame's detection
    list changes between checks.

    Known limitation: two DIFFERENT unknown people who both land in the same
    category (e.g. two separate unrecognized delivery drivers) currently
    share one outcome key, ("unknown", "possible_delivery") -- there's no
    name to distinguish them, so the second one could be suppressed by the
    first one's cooldown. Not solved here; would need a secondary signal
    (e.g. embedding similarity between the two unknown faces) to tell them
    apart.
    """

    def __init__(self, window_size=4, min_hits=2, cooldown_seconds=30):
        self.window_size = window_size          # how many recent checks to remember
        self.min_hits = min_hits                # how many times an outcome must
                                                  # appear within that window to count
        self.cooldown_seconds = cooldown_seconds # minimum gap between repeat
                                                  # announcements of the same outcome

        # Rolling history of the last `window_size` outcomes seen, across
        # ALL detections (a shared timeline, not per-outcome) -- this is what
        # "recent enough" is judged against.
        self._recent = deque(maxlen=window_size)

        # Per-outcome: when was this specific outcome last actually announced.
        # Keyed by outcome tuple, so each person/category tracks independently.
        self._last_announced = {}

    def check(self, outcome):
        """Call once per detection, per check cycle.
        outcome: a hashable identifier for what was detected, e.g.
                 ("known", "anish") or ("unknown", "possible_delivery").
        Returns True if this outcome should be announced right now."""
        now = time.time()

        # Record this outcome as having occurred just now.
        self._recent.append(outcome)

        # Count how many times this specific outcome shows up in the recent
        # window -- tolerant of it not being the literal last entry, since
        # other detections (or brief misses) may have been interleaved.
        hits = list(self._recent).count(outcome)
        if hits < self.min_hits:
            return False  # not seen consistently enough yet -- stay quiet

        # Even if consistent enough, don't repeat the same announcement too
        # soon after the last time it fired.
        last_time = self._last_announced.get(outcome, 0)
        if now - last_time < self.cooldown_seconds:
            return False  # still in cooldown for this specific outcome

        # Clear to announce -- record the time so the cooldown applies going forward.
        self._last_announced[outcome] = now
        return True