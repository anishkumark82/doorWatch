# ring_buffer.py
import os
import time
import cv2


class VisitorRingBuffer:
    """Keeps the last `max_size` unknown-visitor photos on disk.

    No separate index file -- each photo's filename encodes everything
    worth knowing about it (when it was captured, and the specific CLIP
    label guess), so the filesystem itself is the source of truth. This 
    avoids an index ever drifting out of sync with what's actually on 
    disk (a manually deleted photo, a crash mid-write, etc. can't leave 
    stale metadata behind if there's no metadata file to go stale).

    Eviction is trivial: filenames sort chronologically as plain strings
    (since the timestamp is a fixed-width prefix), so the oldest photo is
    always whichever sorts first -- no rename/wraparound logic needed.
    """

    def __init__(self, storage_dir, max_size=50):
        self.storage_dir = storage_dir
        self.max_size = max_size
        os.makedirs(storage_dir, exist_ok=True)

    def add(self, frame, label):
        """Save a photo for an unknown-visitor event. Deletes the oldest
        photo first if the buffer is already at capacity. Returns the
        filename that was written."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{label}.jpg"
        filepath = os.path.join(self.storage_dir, filename)

        # Enforce ring-buffer size by checking what's actually on disk,
        # sorted (chronological, since the timestamp prefix sorts correctly)
        existing = sorted(
            f for f in os.listdir(self.storage_dir) if f.endswith(".jpg")
        )
        if len(existing) >= self.max_size:
            oldest = existing[0]
            os.remove(os.path.join(self.storage_dir, oldest))

        cv2.imwrite(filepath, frame)
        return filename

    def list_photos(self):
        """Return all currently stored photo filenames, oldest first."""
        return sorted(
            f for f in os.listdir(self.storage_dir) if f.endswith(".jpg")
        )