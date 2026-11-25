from insightface.app import FaceAnalysis
import cv2
import numpy as np

def initialize_model():
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=-1,
        det_thresh=.35 # detection threshold -- can be changed although this was able to find a face in each image
    )

    return app


def extract_features(app, image_path):

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    # debug
    if image is None:
        print(f"could not read image {image_path}")
        return 1

    faces = app.get(image)

    if len(faces) == 0:
        print(f"Found 0 faces in {image_path}")
        return 1

    face = max(faces, key=lambda f: f.det_score)

    return face.normed_embedding
    

    


