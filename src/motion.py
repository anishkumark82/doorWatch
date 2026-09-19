# motion.py
import cv2
import numpy as np


class MotionGate:
    """Cheap CPU-only check: is there meaningful pixel change between
    consecutive frames? If not, the caller can skip the expensive
    GPU detection pipeline entirely for this cycle -- most of a door
    camera's day is spent looking at an empty, static scene, so this
    avoids running SCRFD/ArcFace/CLIP when there's nothing to find."""

    def __init__(self, threshold=25, min_changed_fraction=0.01):
        # No previous frame to compare against yet -- set on first call
        self.prev_gray = None

        # How much a single pixel's brightness has to change (0-255 scale)
        # before it counts as "changed" at all. Filters out camera sensor
        # noise and tiny lighting flicker, not just genuine motion.
        self.threshold = threshold

        # What fraction of the ENTIRE frame's pixels need to have changed
        # (by more than `threshold`) before this counts as real motion,
        # rather than a few stray noisy pixels here and there.
        self.min_changed_fraction = min_changed_fraction

    def has_motion(self, frame):
        # Convert to grayscale: motion detection only cares about
        # brightness change, not color -- this is 3x less data to process
        # than the full BGR frame, and simpler to threshold meaningfully.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur before comparing: smooths out sensor noise and small
        # lighting flicker (leaves rustling, a shadow shifting slightly)
        # so only genuinely larger-scale changes register as "motion".
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            # First frame ever seen -- nothing to compare against, so
            # there's no way to know if anything changed. Assume "yes,
            # worth checking" rather than silently skipping the very
            # first cycle after startup.
            self.prev_gray = gray
            return True

        # Absolute difference in brightness, pixel by pixel, between this
        # frame and the last one that was actually compared.
        diff = cv2.absdiff(self.prev_gray, gray)

        # Store this frame as the new baseline for the *next* comparison --
        # comparing against the immediately preceding frame, not a fixed
        # reference image, so gradual lighting changes over the day don't
        # falsely register as constant "motion".
        self.prev_gray = gray

        # Count how many pixels changed by more than `threshold`, as a
        # fraction of the whole frame -- a single flickering pixel won't
        # trigger this, but a person walking through will.
        changed_fraction = np.count_nonzero(diff > self.threshold) / diff.size

        return changed_fraction > self.min_changed_fraction