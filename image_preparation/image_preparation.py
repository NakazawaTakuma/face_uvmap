import cv2
import mediapipe as mp


def image_preparation(image_file, img_size):
    """
    Detects a face in the image, crops the face region with margin,
    resizes it to the desired size, and returns the result.

    Args:
        image_file (str): Path to the input image.
        img_size (int): Desired output size (img_size x img_size).

    Returns:
        np.ndarray or None: Preprocessed face image or None if no face is detected.
    """
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:

        image = cv2.imread(image_file)
        if image is None:
            return None

        imgh, imgw = image.shape[:2]

        # Face detection
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None

        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            xleft = int(box.xmin * imgw)
            xtop = int(box.ymin * imgh)
            xright = int(box.width * imgw + xleft)
            xbottom = int(box.height * imgh + xtop)

            # Expand box
            expansion_rate = 0.3
            expansion_size = int((xright - xleft) * expansion_rate)
            margin_size = [0, 0, 0, 0]
            margin_color = [0, 0, 0]

            xtop -= expansion_size
            if xtop < 0:
                margin_size[0] = -xtop
                xtop = 0

            xbottom += expansion_size
            if xbottom > imgh:
                margin_size[1] = xbottom - imgh
                xbottom = imgh

            xleft -= expansion_size
            if xleft < 0:
                margin_size[2] = -xleft
                xleft = 0

            xright += expansion_size
            if xright > imgw:
                margin_size[3] = xright - imgw
                xright = imgw

            # Crop and pad
            face_box = image[xtop:xbottom, xleft:xright]
            if any(margin_size):
                face_box = cv2.copyMakeBorder(
                    face_box,
                    top=margin_size[0],
                    bottom=margin_size[1],
                    left=margin_size[2],
                    right=margin_size[3],
                    borderType=cv2.BORDER_CONSTANT,
                    value=margin_color,
                )

            # Resize and return
            output_image = cv2.resize(face_box, (img_size, img_size))
            return output_image

        return None  # In case loop doesn't run (shouldn't happen)
