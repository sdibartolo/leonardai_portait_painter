import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# Leonard Phase 1.6A
# MediaPipe Tasks dense facial vision
# ============================================================

CONTOURS = {
    "face_oval": [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ],

    "left_eye": [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246
    ],

    "right_eye": [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398
    ],

    "left_brow": [
        70, 63, 105, 66, 107,
        55, 65, 52, 53, 46
    ],

    "right_brow": [
        300, 293, 334, 296, 336,
        285, 295, 282, 283, 276
    ],

    "outer_lips": [
        61, 146, 91, 181, 84, 17,
        314, 405, 321, 375, 291,
        308, 324, 318, 402, 317,
        14, 87, 178, 88, 95, 78
    ],

    "inner_lips": [
        78, 191, 80, 81, 82, 13,
        312, 311, 310, 415, 308,
        324, 318, 402, 317, 14,
        87, 178, 88, 95
    ],

    "nose": [
        168, 6, 197, 195, 5, 4, 1,
        19, 94, 2, 97, 98, 327, 326,
        129, 49, 279, 358
    ],
}


class PortraitAnalyzer:

    def __init__(self):

        self.backend = "uninitialised"

        # ----------------------------------------------------
        # Locate model
        # ----------------------------------------------------

        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.model_path = os.path.join(
            project_root,
            "models",
            "face_landmarker.task"
        )

        if not os.path.exists(self.model_path):

            raise FileNotFoundError(
                "\n\nLeonard cannot find the MediaPipe Face Landmarker model.\n\n"
                "Expected:\n"
                f"{self.model_path}\n\n"
                "Place face_landmarker.task inside the models folder."
            )

        # ----------------------------------------------------
        # Create MediaPipe Tasks Face Landmarker
        # ----------------------------------------------------

        base_options = python.BaseOptions(
            model_asset_path=self.model_path
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,

            running_mode=vision.RunningMode.IMAGE,

            num_faces=1,

            min_face_detection_confidence=0.35,

            min_face_presence_confidence=0.35,

            min_tracking_confidence=0.35,

            output_face_blendshapes=False,

            output_facial_transformation_matrixes=False,
        )

        self.detector = vision.FaceLandmarker.create_from_options(
            options
        )

        self.backend = "mediapipe-tasks-dense"


    # ========================================================
    # Utility
    # ========================================================

    def _box(self, points, width, height, padding=0.20):

        points = np.asarray(points, dtype=np.float32)

        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)

        bw = max(4, x1 - x0)
        bh = max(4, y1 - y0)

        return (
            max(0, int(x0 - bw * padding)),
            max(0, int(y0 - bh * padding)),

            min(width, int(x1 + bw * padding)),
            min(height, int(y1 + bh * padding)),
        )


    def _centre(self, points):

        p = np.asarray(points, dtype=np.float32)

        return (
            float(p[:, 0].mean()),
            float(p[:, 1].mean()),
        )


    # ========================================================
    # Analyse portrait
    # ========================================================

    def analyze(self, rgb):

        h, w = rgb.shape[:2]

        # MediaPipe expects an mp.Image.

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb)
        )

        result = self.detector.detect(image)

        if not result.face_landmarks:

            raise RuntimeError(
                "\n\nLeonard could not detect a face.\n"
                "Dense facial vision requires a detectable portrait."
            )

        landmarks = result.face_landmarks[0]

        # Convert normalised coordinates -> pixels

        pts = np.array(
            [
                (
                    landmark.x * w,
                    landmark.y * h
                )
                for landmark in landmarks
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Extract semantic contours
        # ----------------------------------------------------

        contours = {}

        for name, indices in CONTOURS.items():

            valid = [
                i for i in indices
                if i < len(pts)
            ]

            contours[name] = [
                tuple(pts[i])
                for i in valid
            ]

        # ----------------------------------------------------
        # Feature regions
        # ----------------------------------------------------

        left_eye_structure = (
            contours["left_eye"]
            + contours["left_brow"]
        )

        right_eye_structure = (
            contours["right_eye"]
            + contours["right_brow"]
        )

        regions = {

            "left_eye":
                self._box(
                    left_eye_structure,
                    w,
                    h,
                    0.30
                ),

            "right_eye":
                self._box(
                    right_eye_structure,
                    w,
                    h,
                    0.30
                ),

            "nose":
                self._box(
                    contours["nose"],
                    w,
                    h,
                    0.35
                ),

            "mouth":
                self._box(
                    contours["outer_lips"],
                    w,
                    h,
                    0.40
                ),

            "jaw":
                self._box(
                    contours["face_oval"],
                    w,
                    h,
                    0.08
                ),

            "face":
                self._box(
                    contours["face_oval"],
                    w,
                    h,
                    0.05
                ),
        }

        # ----------------------------------------------------
        # Face bounding box
        # ----------------------------------------------------

        fx0, fy0, fx1, fy1 = regions["face"]

        face_box = (
            fx0,
            fy0,
            fx1 - fx0,
            fy1 - fy0
        )

        # ----------------------------------------------------
        # Compatibility landmarks
        #
        # painter.py can still use the old keys.
        # ----------------------------------------------------

        semantic_landmarks = {

            "left_eye":
                self._centre(
                    contours["left_eye"]
                ),

            "right_eye":
                self._centre(
                    contours["right_eye"]
                ),

            "nose":
                self._centre(
                    contours["nose"]
                ),

            "mouth":
                self._centre(
                    contours["outer_lips"]
                ),

            "chin":
                tuple(pts[152])
                if len(pts) > 152
                else (
                    (fx0 + fx1) / 2,
                    fy1
                ),
        }

        # ----------------------------------------------------
        # Face mask
        # ----------------------------------------------------

        face_mask = np.zeros(
            (h, w),
            dtype=np.float32
        )

        oval = np.asarray(
            contours["face_oval"],
            dtype=np.int32
        )

        if len(oval) >= 3:

            hull = cv2.convexHull(oval)

            cv2.fillConvexPoly(
                face_mask,
                hull,
                1.0
            )

        # ----------------------------------------------------
        # Structural geometry map
        #
        # This is what painter.py uses to give facial
        # structures additional importance.
        # ----------------------------------------------------

        geometry = np.zeros(
            (h, w),
            dtype=np.float32
        )

        important = [

            "face_oval",

            "left_eye",
            "right_eye",

            "left_brow",
            "right_brow",

            "nose",

            "outer_lips",
            "inner_lips",
        ]

        for name in important:

            points = contours.get(name, [])

            if len(points) < 2:
                continue

            poly = np.asarray(
                points,
                dtype=np.int32
            )

            cv2.polylines(
                geometry,
                [poly],
                True,
                1.0,
                2,
                cv2.LINE_AA
            )

        geometry = cv2.GaussianBlur(
            geometry,
            (0, 0),
            1.4
        )

        if geometry.max() > 0:

            geometry /= geometry.max()

        # ----------------------------------------------------
        # Image edges
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY
        )

        edges = cv2.Canny(
            gray,
            55,
            135
        ).astype(np.float32) / 255.0

        return {

            "face_box":
                face_box,

            "landmarks":
                semantic_landmarks,

            "dense_points":
                pts,

            "contours":
                contours,

            "regions":
                regions,

            "face_mask":
                face_mask,

            "geometry":
                geometry,

            "edges":
                edges,

            "backend":
                self.backend,
        }


    # ========================================================
    # Diagnostic overlay
    # ========================================================

    def overlay(self, rgb, info):

        out = rgb.copy()

        colours = {

            "face_oval":
                (255, 255, 255),

            "left_eye":
                (80, 255, 80),

            "right_eye":
                (80, 255, 80),

            "left_brow":
                (255, 220, 80),

            "right_brow":
                (255, 220, 80),

            "nose":
                (80, 220, 255),

            "outer_lips":
                (255, 100, 180),

            "inner_lips":
                (255, 100, 180),
        }

        # ----------------------------------------------------
        # Draw contours
        # ----------------------------------------------------

        for name, points in info["contours"].items():

            if len(points) < 2:
                continue

            poly = np.asarray(
                points,
                dtype=np.int32
            )

            cv2.polylines(
                out,
                [poly],
                True,
                colours.get(
                    name,
                    (220, 220, 220)
                ),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Draw feature working regions
        # ----------------------------------------------------

        for name, box in info["regions"].items():

            if name == "face":
                continue

            x0, y0, x1, y1 = box

            cv2.rectangle(
                out,
                (x0, y0),
                (x1, y1),
                (200, 200, 200),
                1
            )

            cv2.putText(
                out,
                name.replace("_", " "),
                (
                    x0,
                    max(15, y0 - 4)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Backend indicator
        # ----------------------------------------------------

        cv2.putText(
            out,
            "Vision: " + info["backend"],
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return out