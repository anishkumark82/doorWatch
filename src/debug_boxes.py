# debug_boxes.py
"""Test tool: run SCRFD + ArcFace over saved visitor photos, print every
detection, and write a labeled contact sheet.

  python3 debug_boxes.py                    all photos, real pipeline
  python3 debug_boxes.py 20261004           only files starting with that prefix
  python3 debug_boxes.py 20261004 --no-zone show detections the ignore zone would drop
  python3 debug_boxes.py --thr 0.5          detection threshold (default 0.3, to see weak hits)
"""
import argparse, glob, os, cv2, numpy as np
from trt_infer import TRTEngine
from config import SCRFD_ENGINE, ARCFACE_ENGINE, VISITOR_PHOTOS_DIR
import config
import recognize
from recognize import get_embeddings

ap = argparse.ArgumentParser()
ap.add_argument("prefix", nargs="?", default="")
ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--no-zone", action="store_true",
                help="disable the ignore-zone filter so in-zone detections are visible")
args = ap.parse_args()

zones = list(config.IGNORE_ZONES)
if args.no_zone:
    recognize.IGNORE_ZONES = []   # _in_ignore_zone reads this module-level name

def in_zone(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in zones)

scrfd = TRTEngine(SCRFD_ENGINE)
arcface = TRTEngine(ARCFACE_ENGINE)

paths = sorted(glob.glob(os.path.join(VISITOR_PHOTOS_DIR, args.prefix + "*.jpg")))
tw, th, cols = 480, 270, 5
rows = max(1, (len(paths) + cols - 1) // cols)
sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
no_det = []

for i, p in enumerate(paths):
    img = cv2.imread(p)
    h, w = img.shape[:2]
    sx, sy = tw / w, th / h
    tile = cv2.resize(img, (tw, th))
    name = os.path.basename(p)
    results = get_embeddings(img, scrfd, arcface, score_threshold=args.thr)
    if not results:
        no_det.append(name)
    print(name, "->", len(results), "detection(s)")
    for r in results:
        b = [int(v) for v in r["box"]]
        z = in_zone(b)
        print("    score=%.3f box=%s %s" % (r["score"], b, "IN ZONE" if z else ""))
        color = (0, 0, 255) if z else (0, 255, 0)   # red = in zone, green = elsewhere
        cv2.rectangle(tile, (int(b[0]*sx), int(b[1]*sy)), (int(b[2]*sx), int(b[3]*sy)), color, 2)
    cv2.putText(tile, name[:15], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    r_, c_ = divmod(i, cols)
    sheet[r_*th:(r_+1)*th, c_*tw:(c_+1)*tw] = tile

cv2.imwrite("/tmp/debug_sheet.jpg", sheet)
print("\n%d photos -> /tmp/debug_sheet.jpg" % len(paths))
print("No detection at thr=%.2f: %d" % (args.thr, len(no_det)))
for n in no_det:
    print("   ", n)