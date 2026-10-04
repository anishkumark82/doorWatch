import os, sys
import numpy as np
import cv2
import json
from insightface.utils import face_align

sys.path.append(os.path.dirname(__file__))
from trt_infer import TRTEngine
from utils import bgr_to_model_input
from scrfd_utils import detect_faces, scale_detections
from config import SCRFD_ENGINE, ARCFACE_ENGINE, DB_PATH, IGNORE_ZONES, DETECTION_THRESHOLD
import logging
logger = logging.getLogger("door_watchman")

TEST_IMAGE = os.path.expanduser("~/door-watchman/test_image.jpg")

def _in_ignore_zone(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in IGNORE_ZONES)

def get_embeddings(frame, scrfd, arcface, score_threshold=DETECTION_THRESHOLD):
    """Run the full detect + align + embed pipeline on one frame.
    Returns a list of dicts: {"embedding": 512-d np array, "box": ..., "score": ...}
    (one entry per detected face, using the project's own SCRFD + ArcFace engines)."""

    orig_h, orig_w = frame.shape[:2]

    # Run the scrfd model to determine the area(box) and landmarks of the image where the face appears
    inp = bgr_to_model_input(frame, mean=127.5, std=128.0, size=(640, 640))
    outputs = scrfd.infer(inp)

    detections = detect_faces(outputs, score_threshold=score_threshold)
    if not detections:
        return []

    # scaling the detected area to original scale
    detections = scale_detections(detections, orig_w, orig_h)

    # drop detections centered in a known false-positive zone, but log them
    kept = []
    for d in detections:
        if _in_ignore_zone(d["box"]):
            logger.info(f"Ignored detection in zone: score={d['score']:.3f} "
                        f"box={[int(v) for v in d['box']]}")
        else:
            kept.append(d)
    detections = kept
    if not detections:
        return []

    results = []
    # Run arcface on each detection area
    for det in detections:
        landmarks = np.array(det["landmarks"], dtype=np.float32)
        # Crop the frame based on the landmarks in the detection output. 
        # This will be done in CPU
        aligned = face_align.norm_crop(frame, landmark=landmarks)

        # BGR to RGB conversion
        arcface_inp = bgr_to_model_input(aligned, mean=127.5, std=127.5)
        # Run the arcface model
        outputs_af = arcface.infer(arcface_inp)
        emb = outputs_af[arcface.output_names[0]]
        emb = emb.flatten()
        emb = emb / np.linalg.norm(emb)

        results.append({
            "embedding": emb,
            "box": det["box"],
            "score": det["score"],
        })

    return results

def is_face_usable(frame, box, min_size=60, blur_threshold=50.0):
    """Reject a detected face crop that's too small or too blurry to
    produce a reliable embedding, before spending compute on ArcFace."""
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < min_size or h < min_size:
        return False, "too small"

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "empty crop"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Variance of the Laplacian is a standard, cheap blur-detection metric:
    # a sharp image has high-frequency edges (high variance); a blurry one
    # doesn't. Threshold needs empirical tuning against real captures.
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < blur_threshold:
        return False, f"too blurry ({blur_score:.1f})"

    return True, "ok"

def load_face_db(db_path):
    with open(db_path) as f:
        return json.load(f)

def match_embedding(live_emb, db, threshold=0.5):
    """Linear search: compare live_emb against every stored embedding.
    Returns (name, score) for the best match above threshold, or (None, best_score) if none clear it."""
    best_name = None
    best_score = -1.0

    for name, stored_embeddings in db.items():
        for stored_emb in stored_embeddings:
            stored_emb = np.array(stored_emb, dtype=np.float32)
            score = np.dot(live_emb, stored_emb)  # cosine similarity, both vectors already unit-normalized
            if score > best_score:
                best_score = score
                best_name = name

    if best_score >= threshold:
        return best_name, best_score
    else:
        return None, best_score

if __name__ == "__main__":
    print("Loading SCRFD engine...")
    scrfd = TRTEngine(SCRFD_ENGINE)
    print("Loading ArcFace engine...")
    arcface = TRTEngine(ARCFACE_ENGINE)

    img = cv2.imread(TEST_IMAGE)
    if img is None:
        print(f"Could not read {TEST_IMAGE}")
        sys.exit(1)

    results = get_embeddings(img, scrfd, arcface)
    print(f"\nFound {len(results)} face(s) with embeddings")
    for r in results:
        print(f"  box={r['box']}, det_score={r['score']:.4f}, embedding shape={r['embedding'].shape}")

    db = load_face_db(DB_PATH)
    for r in results:
        name, score = match_embedding(r["embedding"], db)
        if name:
            print(f"  Matched: {name} (similarity={score:.4f})")
        else:
            print(f"  Unknown visitor (best similarity={score:.4f})")