import cv2
import mediapipe as mp


def face_mesh(image):
    """
    Perform face mesh detection on an input image.

    Args:
        image (numpy.ndarray): BGR image.

    Returns:
        tuple:
            - face_landmarks (NormalizedLandmarkList or None): Detected landmarks or None.
            - annotated_image (numpy.ndarray or None): Image with mesh overlay or None.
    """

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        # Convert image to RGB and process
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None, None

        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        return face_landmarks, annotated_image
