import os, sys, json
import numpy as np
import cv2
import insightface
from insightface.utils import face_align

sys.path.append(os.path.dirname(__file__))
from trt_infer import TRTEngine
from utils import bgr_to_model_input
from config import ARCFACE_ENGINE, DB_PATH, PHOTOS_DIR

def main():
    print("Loading detector (CPU, one-time enrollment only)...")
    app = insightface.app.FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1)

    print("Loading ArcFace TensorRT engine...")
    arcface = TRTEngine(ARCFACE_ENGINE)

    db = {}
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            db = json.load(f)
            
    # look through each person directory
    for person in sorted(os.listdir(PHOTOS_DIR)):
        person_dir = os.path.join(PHOTOS_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for fname in sorted(os.listdir(person_dir)):
            path = os.path.join(person_dir, fname)

            # Use open cv to read the image
            img = cv2.imread(path)
            if img is None:
                print(f"  skip unreadable: {fname}")
                continue

            # identify the landmarks in the images using the model.
            faces = app.get(img)
            if not faces:
                print(f"  no face found: {fname}")
                continue
            # Handle cases where more than 1 face was detected
            if len(faces) > 1:
                print(f"  WARNING: {len(faces)} faces found in {fname}, using largest — verify this is correct")
                faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
            face = faces[0]

            # use face_align to normalize the landmarks in the face
            aligned = face_align.norm_crop(img, landmark=face.kps)

            # BGR to RGB conversion
            # ArcFace (aligned 112x112 crop, no resize needed)
            inp = bgr_to_model_input(aligned, mean=127.5, std=127.5)

            #call the inference function using the Tensor RT engine
            outputs = arcface.infer(inp)
            emb = outputs[arcface.output_names[0]] # output in (1, 512) - 2D array
            emb = emb.flatten() # now (512,) - 1D array

            # L2 normalization
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb.tolist())
            print(f"  enrolled: {fname}")

        if embeddings:
            db[person] = embeddings
            print(f"{person}: {len(embeddings)} embedding(s) stored")

    #Create JSON file for faces
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(db, f)
    print(f"Saved to {DB_PATH}")

if __name__ == "__main__":
    main()
