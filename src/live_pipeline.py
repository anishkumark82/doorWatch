import os, sys, time, threading
import cv2

sys.path.append(os.path.dirname(__file__))
from trt_infer import TRTEngine
from motion import MotionGate
from config import (SCRFD_ENGINE, ARCFACE_ENGINE, DB_PATH, GO2RTC_STREAM, CLIP_ENGINE, 
                    CLIP_TEXT_EMBEDDINGS_PATH, VISITOR_PHOTOS_DIR, IDLE_INTERVAL, 
                    ACTIVE_INTERVAL, MOTION_THRESHOLD, MOTION_MIN_CHANGED_FRACTION, 
                    FORCE_CHECK_EVERY, LOG_PATH, LOG_IDLE_EVERY, GATE_COOLDOWN_SECONDS, DETECTION_THRESHOLD)
from recognize import get_embeddings, load_face_db, match_embedding
from classify import DeliveryClassifier
from notify import build_announcement, AnnouncementGate, speak_announcement, send_push_notification
from ring_buffer import VisitorRingBuffer
from logging_setup import get_logger, ThrottledLogger

# retrive the logger
logger = get_logger(LOG_PATH)
idle_logger = ThrottledLogger(logger, interval=LOG_IDLE_EVERY)

class FrameGrabber:
    """Continuously decodes frames in a background thread, always keeping
    only the most recent one. The main loop never waits on a growing
    backlog -- it just reads whatever's freshest whenever it's ready.

    Also self-heals if the stream connection was never established or drops:
    OpenCV's RTSP backend does not reliably retry a failed connection on its
    own, so after reconnect_interval seconds with no successful frame, the
    VideoCapture object is torn down and recreated from scratch."""

    def __init__(self, url, reconnect_interval=5.0):
        self.url = url
        self.reconnect_interval = reconnect_interval
        self.cap = cv2.VideoCapture(url)
        self.latest_frame = None
        # Locking access to the latest frame.
        self.lock = threading.Lock()
        self.running = True
        # Reader thread reading the video capture frame
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        # Tracks the last time a frame was successfully read -- used to
        # detect a stuck/dead connection that needs to be recreated.
        last_success = time.time()
        while self.running:
            # reading the captured frame
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
                last_success = time.time()
            else:
                # No successful read for a while -- assume the connection is
                # stuck/dead (e.g. go2rtc wasn't up yet, or it restarted) and
                # recreate it, rather than looping forever on a VideoCapture
                # that has already proven it won't recover on its own.
                if time.time() - last_success > self.reconnect_interval:
                    print(f"[FrameGrabber] No frames for {self.reconnect_interval}s, reconnecting to {self.url}...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.url)
                    last_success = time.time()  # reset the clock after a reconnect attempt
                time.sleep(0.5)  # avoid a tight spin loop while disconnected

    def read(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0) # wait for _reader() to actually exit its loop
        self.cap.release() # onl release once the thread is done using it

def main():
    # ---- Load all the model engines -----
    logger.info("Loading SCRFD engine...")
    scrfd = TRTEngine(SCRFD_ENGINE)

    logger.info("Loading ArcFace engine...")
    arcface = TRTEngine(ARCFACE_ENGINE)

    logger.info("Loading clip classifier...")
    clip_classifier = DeliveryClassifier(CLIP_ENGINE, CLIP_TEXT_EMBEDDINGS_PATH)

    logger.info("Loading enrolled face database...")
    db = load_face_db(DB_PATH)
    logger.info(f"  {len(db)} people enrolled: {list(db.keys())}")

    logger.info(f"Connecting to {GO2RTC_STREAM}...")
    grabber = FrameGrabber(GO2RTC_STREAM)

    logger.info("Setting up announcement debouncing ...")
    gate = AnnouncementGate(window_size=4, min_hits=2, cooldown_seconds=GATE_COOLDOWN_SECONDS)

    visitor_buffer = VisitorRingBuffer(VISITOR_PHOTOS_DIR, max_size=50)
    logger.info("Create the Visitor Buffer instance ....")

    motion_gate = MotionGate(threshold=MOTION_THRESHOLD, min_changed_fraction=MOTION_MIN_CHANGED_FRACTION)
    current_interval = IDLE_INTERVAL
    last_forced_check = 0
    logger.info("Starting motion detection logic instance ....")

    logger.info("Starting live recognition loop. Ctrl+C to stop.\n")
    printed_shape = False

    try:
        while True:
            frame = grabber.read()
            if frame is None:
                logger.info("Waiting for first frame...")
                time.sleep(0.5)
                continue
            if not printed_shape:
                logger.info(f"Frame shape: {frame.shape}")
                printed_shape = True

            now = time.time()
            # Check if there is any motion detected 
            motion = motion_gate.has_motion(frame)
            # if idle is it beyond the FORCE_CHECK_TIME (10sec) ?
            force_check = (now - last_forced_check) >= FORCE_CHECK_EVERY

            # No motion and no force check avoid running models in GPU
            if not motion and not force_check:
                idle_logger.info("Idle -- No motion")
                current_interval = IDLE_INTERVAL
                time.sleep(current_interval)
                continue
            # Check and retrieve embeddings and set the cuttent time to track last embedding time
            last_forced_check = now

            # Get embeddings for the current image 
            # 1. Run Scarfd [3 scales] to determine faces [boxes, landmarks]
            # 2. Run arcface to get the embeddings 
            results = get_embeddings(frame, scrfd, arcface, score_threshold=DETECTION_THRESHOLD)

            if results:
                outcomes_this_frame = []
                current_interval = ACTIVE_INTERVAL
                for r in results:
                    ts = time.strftime('%H:%M:%S')
                    # Compare with face.json to determine if there is matching embeddings    
                    name, score = match_embedding(r["embedding"], db)
                    
                    if name:
                        outcome = ("known", name)
                        text = build_announcement(name=name)
                        logger.info(f"Known: {name} (similarity={score:.4f})-> \"{text}\"")
                    else:
                        # Unknown and try with clip
                        label, clip_score = clip_classifier.classify(frame, r["box"])
                        outcome = ("unknown", label)
                        text = build_announcement(label=label)

                        logger.info(f"Unknown visitor (best face similarity={score:.4f}) "
                              f"-- (specific guess: {label}, score={clip_score:.4f})-> \"{text}\"")

                    outcomes_this_frame.append(outcome)
                    if gate.check(outcome):
                        logger.info(f">>> ANNOUNCE: \"{text}\"")
                        speak_announcement(text)
                        send_push_notification(text)
                        if not name:  # only save photos for unknown visitors
                            saved_as = visitor_buffer.add(frame, label)
                            logger.info(f"<<< saved: {saved_as}")
                    else:
                        logger.info(f"*** (suppressed)")
                assert len(outcomes_this_frame) == len(results), \
                        f"Mismatch: {len(results)} detections but {len(outcomes_this_frame)} outcomes"
            else:
                idle_logger.info(f"Idle -- No face detected")
                current_interval = IDLE_INTERVAL
            time.sleep(current_interval)

    except KeyboardInterrupt:
        logger.info(".... Stopping ....")
        grabber.stop()

if __name__ == "__main__":
    main()