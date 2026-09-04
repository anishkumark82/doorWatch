import os, sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(__file__))
from trt_infer import TRTEngine
from utils import bgr_to_model_input
from scrfd_utils import detect_faces, scale_detections
from config import SCRFD_ENGINE

TEST_IMAGE = os.path.expanduser("~/door-watchman/test_image.jpg")

def main():
    print("Loading SCRFD TensorRT engine...")
    scrfd = TRTEngine(SCRFD_ENGINE)

    img = cv2.imread(TEST_IMAGE)
    if img is None:
        print(f"Could not read {TEST_IMAGE}")
        return
    print(f"Input image shape: {img.shape}")

    inp = bgr_to_model_input(img, mean=127.5, std=128.0, size=(640, 640))
    print(f"Preprocessed input shape: {inp.shape}")

    # Call the TensorRT engine for generating scrfd output
    outputs = scrfd.infer(inp)

    # Function to detect faces using the output in 640x640
    detections = detect_faces(outputs)
    
    # Scale the boxes to original size
    orig_h, orig_w = img.shape[:2]
    dections_scaled = scale_detections(detections, orig_w, orig_h)

    print(f"Found {len(dections_scaled)} face(s)")
    for d in dections_scaled:
        print(f"  box={d['box']}, score={d['score']:.4f}")
        print(f"  landmarks={d['landmarks']}")

if __name__ == "__main__":
    main()