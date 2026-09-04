# utils.py
import cv2
import numpy as np

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def bgr_to_model_input(img_bgr, mean, std, size=None):
    img = img_bgr
    if size is not None:
        img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return np.ascontiguousarray(img)


def bgr_to_clip_input(img_bgr, size=(224, 224)):
    img = cv2.resize(img_bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img / 255.0
    img = (img - CLIP_MEAN) / CLIP_STD
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return np.ascontiguousarray(img)