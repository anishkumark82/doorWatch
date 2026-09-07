import os, sys, time, threading
import cv2

sys.path.append(os.path.dirname(__file__))
from trt_infer import TRTEngine
from config import SCRFD_ENGINE, ARCFACE_ENGINE, DB_PATH, GO2RTC_STREAM, CHECK_INTERVAL, CLIP_ENGINE, CLIP_TEXT_EMBEDDINGS_PATH, VISITOR_PHOTOS_DIR
from recognize import get_embeddings, load_face_db, match_embedding, is_face_usable
from classify import DeliveryClassifier
from notify import build_announcement, AnnouncementGate, speak_announcement, send_push_notification
from ring_buffer import VisitorRingBuffer

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
    print("Loading SCRFD engine...")
    scrfd = TRTEngine(SCRFD_ENGINE)

    print("Loading ArcFace engine...")
    arcface = TRTEngine(ARCFACE_ENGINE)

    print("Loading clip classifier...")
    clip_classifier = DeliveryClassifier(CLIP_ENGINE, CLIP_TEXT_EMBEDDINGS_PATH)

    print("Loading enrolled face database...")
    db = load_face_db(DB_PATH)
    print(f"  {len(db)} people enrolled: {list(db.keys())}")

    print(f"Connecting to {GO2RTC_STREAM}...")
    grabber = FrameGrabber(GO2RTC_STREAM)

    print("Setting up announcement debouncing ...")
    gate = AnnouncementGate(window_size=4, min_hits=2, cooldown_seconds=30)

    visitor_buffer = VisitorRingBuffer(VISITOR_PHOTOS_DIR, max_size=50)
    print("Create the Visitor Buffer instance ....")

    print("Starting live recognition loop. Ctrl+C to stop.\n")
    printed_shape = False

    try:
        while True:
            frame = grabber.read()
            if frame is None:
                print("Waiting for first frame...")
                time.sleep(0.5)
                continue
            if not printed_shape:
                print(f"Frame shape: {frame.shape}")
                printed_shape = True

            # Get embeddings for the current image 
            # 1. Run Scarfd [3 scales] to determine faces [boxes, landmarks]
            # 2. Run arcface to get the embeddings 
            results = get_embeddings(frame, scrfd, arcface)

            if results:
                outcomes_this_frame = []
                for r in results:
                    ts = time.strftime('%H:%M:%S')
                    
                    # Compare with face.json to determine if there is matching embeddings    
                    name, score = match_embedding(r["embedding"], db)
                    
                    if name:
                        outcome = ("known", name)
                        text = build_announcement(name=name)
                        print(f"[{ts}] Known: {name} (similarity={score:.4f})-> \"{text}\"")
                    else:
                        # Unknown and try with clip
                        category, label, clip_score = clip_classifier.classify_category(frame, r["box"])
                        outcome = ("unknown", category)
                        text = build_announcement(category=category, label=label)

                        print(f"[{ts}] Unknown visitor (best face similarity={score:.4f}) "
                              f"-- category={category} (specific guess: {label}, score={clip_score:.4f})-> \"{text}\"")

                    outcomes_this_frame.append(outcome)
                    if gate.check(outcome):
                        print(f"[{ts}] >>> ANNOUNCE: \"{text}\"")
                        speak_announcement(text)
                        send_push_notification(text)
                        if not name:  # only save photos for unknown visitors
                            saved_as = visitor_buffer.add(frame, category, label)
                            print(f"[{ts}]     saved: {saved_as}")
                    else:
                        print(f"[{ts}]     (suppressed)")
                assert len(outcomes_this_frame) == len(results), \
                        f"Mismatch: {len(results)} detections but {len(outcomes_this_frame)} outcomes"
            else:
                print(f"[{time.strftime('%H:%M:%S')}] No face detected", end="\r")
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopping...")
        grabber.stop()

if __name__ == "__main__":
    main()