# scrfd_utils.py
import numpy as np

# SCRFD detects at 3 scales; stride/tensor names differ per scale but the
# thresholding/decoding logic (process_scale) is identical for all three
SCALES = [
    {"score_key": "448", "box_key": "451", "landmark_key": "454", "stride": 8},
    {"score_key": "471", "box_key": "474", "landmark_key": "477", "stride": 16},
    {"score_key": "494", "box_key": "497", "landmark_key": "500", "stride": 32},
]

def decode_bbox(anchor_center, raw_box, stride):
    cx, cy = anchor_center
    left, top, right, bottom = raw_box * stride
    x1 = cx - left
    y1 = cy - top
    x2 = cx + right
    y2 = cy + bottom
    return x1, y1, x2, y2


def decode_landmarks(anchor_center, raw_landmarks, stride):
    cx, cy = anchor_center
    points = []
    for i in range(5):
        dx = raw_landmarks[i*2] * stride
        dy = raw_landmarks[i*2 + 1] * stride
        points.append((cx + dx, cy + dy))
    return points


def process_scale(outputs, score_key, box_key, stride, threshold=0.5):
    """Threshold + decode boxes for one SCRFD detection scale.
    Returns (decoded_boxes, scores) for candidates that survived thresholding."""
    #flatten scores
    scores = outputs[score_key].flatten()

    #Apply thresholding
    keep_idx = np.where(scores > threshold)[0]

    if len(keep_idx) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    # the standard size for SCRFD input feed is 640x640
    # stride varies by Scale 1,2 or 3
    boxes_raw = outputs[box_key]
    grid_size = 640 // stride

    #determine the boxes in where the face detection is within threshold
    decoded = []
    for idx in keep_idx:
        location = idx // 2
        row = location // grid_size
        col = location % grid_size
        anchor_center = (col * stride, row * stride)
        decoded.append(decode_bbox(anchor_center, boxes_raw[idx], stride))

    return np.array(decoded), scores[keep_idx], keep_idx


def compute_iou(box, boxes):
    # Find the overlapping rectangle between `box` and each box in `boxes`.
    # Top-left corner of the overlap: the *larger* of the two top-left corners
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    # Bottom-right corner of the overlap: the *smaller* of the two bottom-right corners
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # Width/height of the overlap region. If the boxes don't actually overlap,
    # x2-x1 or y2-y1 would go negative -- clamp to 0 so "no overlap" = 0 area,
    # not a nonsensical negative area.
    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    intersection = inter_w * inter_h

    # Area of each box individually
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Union = both areas combined, minus the overlap counted twice
    union = area_box + area_boxes - intersection

    # IoU = overlap area / combined area -> 0 (no overlap) to 1 (identical boxes)
    # 1e-6 floor avoids a divide-by-zero if a box has zero area
    return intersection / np.maximum(union, 1e-6)


def nms(boxes, scores, iou_threshold=0.4):
    # Process candidates highest-confidence first
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        # The current highest-scoring remaining candidate is always kept --
        # it's the "maximum" that gives Non-Maximum Suppression its name
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Compare every other remaining candidate's box against the one just kept
        rest = indices[1:]
        ious = compute_iou(boxes[current], boxes[rest])

        # Keep only the candidates that DON'T overlap much with the one just kept
        # (low IoU = probably a different, real, separate face)
        # Anything overlapping heavily (high IoU) is dropped as a duplicate
        # detection of the same face and never considered again
        indices = rest[ious < iou_threshold]

    return keep

def detect_faces(outputs, score_threshold=0.5, iou_threshold=0.4):
    """Full SCRFD post-processing: raw engine output -> final face detections.
    Returns a list of dicts: {"box": (x1,y1,x2,y2), "landmarks": [...], "score": float}
    Coordinates are in 640x640 model-input space."""
    all_boxes = []
    all_scores = []
    all_keep_idx = []
    all_strides = []

    # 1. Threshold + decode boxes, per scale
    for scale in SCALES:
        boxes, scores, keep_idx = process_scale(
            outputs, scale["score_key"], scale["box_key"], scale["stride"], score_threshold
        )
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_keep_idx.extend(keep_idx)
        all_strides.extend([scale["stride"]] * len(keep_idx))

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    if len(all_boxes) == 0:
        return []

    # 2. Collapse duplicate detections of the same face across all scales combined
    nms_keep = nms(all_boxes, all_scores, iou_threshold)

    # 3. Decode landmarks only for the final survivors (cheaper than decoding
    # landmarks for every candidate before filtering)
    detections = []
    for i in nms_keep:
        idx = all_keep_idx[i]
        stride = all_strides[i]
        grid_size = 640 // stride
        location = idx // 2
        row = location // grid_size
        col = location % grid_size
        anchor_center = (col * stride, row * stride)

        scale_entry = next(s for s in SCALES if s["stride"] == stride)
        landmarks_raw = outputs[scale_entry["landmark_key"]][idx]
        landmarks = decode_landmarks(anchor_center, landmarks_raw, stride)

        detections.append({
            "box": tuple(all_boxes[i]),
            "landmarks": landmarks,
            "score": float(all_scores[i]),
        })

    return detections

def scale_detections(detections, orig_w, orig_h, model_size=640):
    """Function to scale back the detection boxes/landmarks to the original size"""
    if not detections:
        return []

    scale_x = orig_w / model_size
    scale_y = orig_h / model_size

    scaled = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        box = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

        landmarks = [(px * scale_x, py * scale_y) for px, py in det["landmarks"]]

        scaled.append({
            "box": box,
            "landmarks": landmarks,
            "score": det["score"],
        })
    return scaled