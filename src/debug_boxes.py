# review_all.py - file to test false detections
import glob, os, cv2, numpy as np
from trt_infer import TRTEngine
from config import SCRFD_ENGINE, ARCFACE_ENGINE, VISITOR_PHOTOS_DIR
from recognize import get_embeddings

ZONE = (1150, 0, 1800, 450)   # candidate ignore zone (x1, y1, x2, y2)

def in_zone(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return ZONE[0] <= cx <= ZONE[2] and ZONE[1] <= cy <= ZONE[3]

scrfd = TRTEngine(SCRFD_ENGINE)
arcface = TRTEngine(ARCFACE_ENGINE)

paths = sorted(glob.glob(os.path.join(VISITOR_PHOTOS_DIR, "*.jpg")))
tw, th, cols = 480, 270, 5
rows = (len(paths) + cols - 1) // cols
sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
no_det, outside, inzone = [], [], []

for i, p in enumerate(paths):
    img = cv2.imread(p)
    h, w = img.shape[:2]
    sx, sy = tw / w, th / h
    tile = cv2.resize(img, (tw, th))
    results = get_embeddings(img, scrfd, arcface, score_threshold=0.3)

    name = os.path.basename(p)
    if not results:
        no_det.append(name)
    for r in results:
        b = [int(v) for v in r["box"]]
        z = in_zone(b)
        entry = (name, round(float(r["score"]), 3), b)
        (inzone if z else outside).append(entry)
        color = (0, 0, 255) if z else (0, 255, 0)   # red = in zone, green = elsewhere
        cv2.rectangle(tile, (int(b[0]*sx), int(b[1]*sy)), (int(b[2]*sx), int(b[3]*sy)), color, 2)

    cv2.putText(tile, name[:15], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    r_, c_ = divmod(i, cols)
    sheet[r_*th:(r_+1)*th, c_*tw:(c_+1)*tw] = tile

cv2.imwrite("/tmp/all_photos_sheet.jpg", sheet)
print(len(paths), "photos -> /tmp/all_photos_sheet.jpg")

print("\nIn-zone detections from 09-27 (the ones that matter):")
hits = [e for e in inzone if e[0].startswith("20260927")]
for n, s, b in hits:
    print("  ", n, "score=", s, "box=", b)
if not hits:
    print("   none")

print("\nIn-zone detections from 09-28: %d (the car)" % sum(1 for e in inzone if e[0].startswith("20260928")))

print("\nPhotos with NO detection at 0.3:")
for n in no_det:
    print("  ", n)