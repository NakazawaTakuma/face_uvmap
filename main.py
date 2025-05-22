import numpy as np
import cv2
from pathlib import Path
import os

# ImagePreparation
from image_preparation.image_preparation import image_preparation

# FaceMesh
from face_mesh.face_mesh import face_mesh

# MakeUV
from uv_generator.make_uv_main import generate_uvmap

from config import IMAGE_FOLDER, SAVE_FOLDER, IMG_SIZE

SAVE_UV_FOLDER = os.path.join(SAVE_FOLDER, "uv")

def ensure_directories():
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    os.makedirs(SAVE_UV_FOLDER, exist_ok=True)

def process_image(path):
    print(f"Processing: {path}")
    image = image_preparation(str(path), IMG_SIZE)
    if image is None or not image.any():
        print("Skipping invalid image.")
        return

    # Face mesh
    face_landmarks, annotated_image = face_mesh(image)
    if face_landmarks is None:
        print("Skipping image without landmarks.")
        return

    # Save annotated face image
    cv2.imwrite(os.path.join(SAVE_FOLDER, path.name), annotated_image)

    # UV map generation
    uv_image = generate_uvmap(image, face_landmarks, IMG_SIZE)
    uv_image = np.array(uv_image, dtype=np.uint8)
    cv2.imwrite(os.path.join(SAVE_UV_FOLDER, path.name), uv_image)

   

def get_image_paths(folder, extensions=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]):
    pathlist = []
    for ext in extensions:
        pathlist.extend(Path(folder).rglob(ext))
    return pathlist

def main():
    ensure_directories()
    pathlist = get_image_paths(IMAGE_FOLDER)

    for path in pathlist:
        process_image(path)

if __name__ == "__main__":
    main()
