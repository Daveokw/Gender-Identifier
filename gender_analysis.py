"""Reliable facial-presentation analysis for the Streamlit interface."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MAX_IMAGE_PIXELS = 24_000_000
MIN_FACE_EDGE = 64
BLUR_EVALUATION_EDGE = 256
MIN_BLUR_SCORE = 8.0
MIN_FACE_QUALITY_SCORE = 0.30
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 225.0
MIN_CONTRAST = 18.0


@dataclass(frozen=True)
class GenderPrediction:
    """A prediction that passed every image-quality check."""

    label: str


@dataclass(frozen=True)
class AnalysisOutcome:
    """Either a prediction or an actionable reason why one was not made."""

    prediction: GenderPrediction | None = None
    message: str | None = None


def prepare_image(image: Image.Image) -> tuple[np.ndarray, float]:
    """Return a bounded BGR image and its scale relative to the upload."""

    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("The image has invalid dimensions.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError("The image is too large. Please upload a smaller copy.")

    rgb = np.asarray(image)
    longest_edge = max(width, height)
    scale = 1.0

    # Moderate upscaling helps the detector find distant faces. The original
    # face dimensions are still checked later, so this cannot invent detail.
    if longest_edge < 1_200:
        scale = min(2.0, 1_200 / longest_edge)
    elif longest_edge > 3_000:
        scale = 3_000 / longest_edge

    if scale != 1.0:
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        rgb = cv2.resize(
            rgb, dsize=None, fx=scale, fy=scale, interpolation=interpolation
        )

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), scale


@lru_cache(maxsize=1)
def _load_models() -> tuple[Any, Any, Any]:
    """Load shared detection, quality, and classification models."""

    from uniface.attribute import FairFace
    from uniface.constants import RetinaFaceWeights
    from uniface.detection import RetinaFace
    from uniface.quality import EDifFIQA

    providers = ["CPUExecutionProvider"]
    detector = RetinaFace(
        model_name=RetinaFaceWeights.RESNET34,
        confidence_threshold=0.75,
        providers=providers,
    )
    quality_model = EDifFIQA(providers=providers)
    classifier = FairFace(providers=providers)
    return detector, quality_model, classifier


def analyse_gender(
    image: Image.Image,
    detector: Any | None = None,
    quality_model: Any | None = None,
    classifier: Any | None = None,
) -> AnalysisOutcome:
    """Analyse one usable face with RetinaFace, eDifFIQA, and FairFace."""

    try:
        analysed_image, scale = prepare_image(image)
    except (OSError, ValueError) as exc:
        return AnalysisOutcome(message=str(exc))

    try:
        if detector is None or quality_model is None or classifier is None:
            default_detector, default_quality_model, default_classifier = _load_models()
            detector = detector or default_detector
            quality_model = quality_model or default_quality_model
            classifier = classifier or default_classifier

        faces = detector.detect(analysed_image)
        face_issue = _check_detected_faces(faces)
        if face_issue:
            return AnalysisOutcome(message=face_issue)

        face = faces[0]
        quality_issue = _check_face_quality(face, analysed_image, scale)
        if quality_issue:
            return AnalysisOutcome(message=quality_issue)

        quality_result = quality_model.predict(analysed_image, face.landmarks)
        quality_score = float(getattr(quality_result, "score", float("nan")))
        if not np.isfinite(quality_score):
            logger.error("The face-quality model returned an invalid score")
            return AnalysisOutcome(message="The image could not be analysed reliably.")
        if quality_score < MIN_FACE_QUALITY_SCORE:
            return AnalysisOutcome(
                message="The face lacks enough usable detail. Try a clearer, closer photo."
            )

        result = classifier.predict(analysed_image, face)
        label = getattr(result, "sex", None)
        if label not in {"Male", "Female"}:
            logger.error("FairFace returned an unsupported label: %r", label)
            return AnalysisOutcome(message="The image could not be analysed reliably.")
    except Exception:
        logger.exception("Face analysis failed")
        return AnalysisOutcome(
            message="Analysis could not be completed. Please try another image."
        )

    return AnalysisOutcome(prediction=GenderPrediction(label=label))


def _check_detected_faces(faces: Any) -> str | None:
    if not isinstance(faces, Sequence):
        return "The face detector returned an unexpected result."
    if len(faces) == 0:
        return "No clear face was detected. Try a closer, sharper, front-facing photo."
    if len(faces) > 1:
        return "Multiple faces were detected. Use an image containing one clearly visible person."
    return None


def _check_face_quality(
    face_result: Any,
    analysed_image: np.ndarray,
    scale: float,
) -> str | None:
    bbox = getattr(face_result, "bbox", None)
    if bbox is None:
        return "The detected face could not be measured reliably."

    try:
        x1, y1, x2, y2 = (int(value) for value in bbox[:4])
    except (TypeError, ValueError):
        return "The detected face could not be measured reliably."

    image_height, image_width = analysed_image.shape[:2]
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return "The detected face could not be measured reliably."
    if min(width, height) / scale < MIN_FACE_EDGE:
        return "The face is too far from the camera. Upload a closer portrait or crop the image."

    face = analysed_image[y1:y2, x1:x2]
    if face.size == 0:
        return "The detected face could not be measured reliably."

    grey_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    if _blur_score(grey_face) < MIN_BLUR_SCORE:
        return "The face appears blurred. Use a sharper photo with less movement."

    brightness = float(grey_face.mean())
    if brightness < MIN_BRIGHTNESS:
        return "The face is too dark. Use a brighter, evenly lit photo."
    if brightness > MAX_BRIGHTNESS:
        return "The face is overexposed. Use a photo with softer, even lighting."
    if float(grey_face.std()) < MIN_CONTRAST:
        return "The facial details have too little contrast. Try a clearer, better-lit photo."

    return None


def _blur_score(grey_face: np.ndarray) -> float:
    """Measure sharpness at a stable size and reject only severely blurred faces."""

    height, width = grey_face.shape
    scale = BLUR_EVALUATION_EDGE / max(height, width)
    if scale != 1.0:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        grey_face = cv2.resize(
            grey_face,
            dsize=None,
            fx=scale,
            fy=scale,
            interpolation=interpolation,
        )

    return float(cv2.Laplacian(grey_face, cv2.CV_64F).var())
