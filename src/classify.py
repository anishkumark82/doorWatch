import os, sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(__file__))
from utils import bgr_to_clip_input
from config import CLIP_TEXT_EMBEDDINGS_PATH
from trt_infer import TRTEngine

DELIVERY_CATEGORY_MAP = {
    "amazon_delivery": "possible_delivery",
    "fedex_delivery": "possible_delivery",
    "ups_delivery": "possible_delivery",
    "food_delivery": "possible_delivery",
    "food_delivery_branded": "possible_delivery",
    "regular_visitor": "regular_visitor",
}
class DeliveryClassifier:
    def __init__(self, clip_engine_path, text_embeddings_path):
        self.engine = TRTEngine(clip_engine_path)
        self.labels, self.text_embeddings = self._load_clip_text_embeddings(text_embeddings_path)

    @staticmethod
    def _load_clip_text_embeddings(path):
        data = np.load(path)
        return list(data["labels"]), data["embeddings"]

    def crop_person_region(self, frame, box, pad_x=100, pad_y=200):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y // 4)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        return frame[y1:y2, x1:x2]

    def classify(self, frame, box):
        region = self.crop_person_region(frame, box)
        inp = bgr_to_clip_input(region, size=(224, 224))
        outputs = self.engine.infer(inp)
        img_emb = outputs[self.engine.output_names[0]]
        img_emb = img_emb.flatten()
        img_emb = img_emb / np.linalg.norm(img_emb)

        scores = self.text_embeddings @ img_emb
        best_idx = np.argmax(scores)
        return self.labels[best_idx], float(scores[best_idx])

    def classify_category(self, frame, box):
        """Same as classify(), but returns the collapsed category
        (possible_delivery / regular_visitor) rather than the specific brand label."""
        label, score = self.classify(frame, box)
        category = DELIVERY_CATEGORY_MAP[label]
        return category, label, score

if __name__ == "__main__":
    from config import CLIP_ENGINE, CLIP_TEXT_EMBEDDINGS_PATH

    TEST_IMAGE = os.path.expanduser("~/door-watchman/test_image_clip2.jpg")

    print("Loading DeliveryClassifer for clip...")
    classifier = DeliveryClassifier(CLIP_ENGINE, CLIP_TEXT_EMBEDDINGS_PATH)
    print(f"  {len(classifier.labels)} categories: {classifier.labels}")

    img = cv2.imread(TEST_IMAGE)
    if img is None:
        print(f"Could not read {TEST_IMAGE}")
        sys.exit(1)
    # determine the width and ht
    h, w = img.shape[:2]
    full_box = (0, 0, w, h)
    #label, score = classifier.classify(img, full_box)
    category, label, score = classifier.classify_category(img, full_box)
    print(f"\nBest match: {label} (score={score:.4f}) Category : {category}")