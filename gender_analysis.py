"""Reliable facial-presentation analysis for the Streamlit interface."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MAX_IMAGE_PIXELS = 24_000_000
MIN_FACE_EDGE = 64
MIN_DETECTION_CONFIDENCE = 0.90
MIN_CLASSIFICATION_CONFIDENCE = 75.0
MIN_CLASSIFICATION_MARGIN = 25.0
MIN_BLUR_SCORE = 55.0
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 225.0
MIN_CONTRAST = 18.0


@dataclass(frozen=True)
class GenderPrediction:
    """A prediction that passed every reliability threshold."""

    label: str
    confidence: float
    detection_confidence: float


@dataclass(frozen=True)
class AnalysisOutcome:
    """Either a reliable prediction or an actionable abstention message."""

    prediction: GenderPrediction | None = None
    message: str | None = None


Analyser = Callable[..., Any]


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

    # Upscaling can help RetinaFace locate a distant face, but the original
    # face size is still checked later so interpolation cannot create certainty.
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


def interpret_result(
    raw_result: Any,
    analysed_image: np.ndarray,
    scale: float,
) -> AnalysisOutcome:
    """Apply quality and confidence gates to one DeepFace response."""

    if isinstance(raw_result, Mapping):
        results = [raw_result]
    elif isinstance(raw_result, Sequence) and not isinstance(raw_result, (str, bytes)):
        results = raw_result
    else:
        return AnalysisOutcome(
            message="The face analyser returned an unexpected result."
        )
    if len(results) == 0:
        return AnalysisOutcome(
            message="No face was detected. Try a clearer or closer photo."
        )
    if len(results) > 1:
        return AnalysisOutcome(
            message="Multiple faces were detected. Use an image containing one clearly visible person."
        )

    result = results[0]
    if not isinstance(result, Mapping):
        return AnalysisOutcome(
            message="The face analyser returned an unexpected result."
        )

    quality_issue = _check_face_quality(result, analysed_image, scale)
    if quality_issue:
        return AnalysisOutcome(message=quality_issue)

    gender_scores = result.get("gender")
    if not isinstance(gender_scores, Mapping):
        return AnalysisOutcome(
            message="The model did not return usable classification scores."
        )

    try:
        scores = {
            "Man": float(gender_scores["Man"]),
            "Woman": float(gender_scores["Woman"]),
        }
    except (KeyError, TypeError, ValueError):
        return AnalysisOutcome(
            message="The model did not return usable classification scores."
        )
    if any(
        not math.isfinite(score) or not 0.0 <= score <= 100.0
        for score in scores.values()
    ):
        return AnalysisOutcome(
            message="The model did not return usable classification scores."
        )

    ranked_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    dominant_label, confidence = ranked_scores[0]
    margin = confidence - ranked_scores[1][1]

    if confidence < MIN_CLASSIFICATION_CONFIDENCE or margin < MIN_CLASSIFICATION_MARGIN:
        return AnalysisOutcome(
            message=(
                "The model is not confident enough to provide a reliable estimate. "
                "Try a front-facing portrait with the face closer to the camera."
            )
        )

    display_label = "Male" if dominant_label == "Man" else "Female"
    return AnalysisOutcome(
        prediction=GenderPrediction(
            label=display_label,
            confidence=confidence,
            detection_confidence=float(result.get("face_confidence", 0.0)),
        )
    )


def analyse_gender(
    image: Image.Image, analyser: Analyser | None = None
) -> AnalysisOutcome:
    """Analyse an image, abstaining whenever the result is not reliable."""

    try:
        analysed_image, scale = prepare_image(image)
    except (OSError, ValueError) as exc:
        return AnalysisOutcome(message=str(exc))

    if analyser is None:
        from deepface import DeepFace

        analyser = DeepFace.analyze

    try:
        raw_result = analyser(
            img_path=analysed_image,
            actions=["gender"],
            enforce_detection=True,
            detector_backend="retinaface",
            align=True,
            expand_percentage=10,
            silent=True,
        )
    except ValueError as exc:
        if "face could not be detected" in str(exc).casefold():
            return AnalysisOutcome(
                message="No clear face was detected. Try a closer, sharper, front-facing photo."
            )
        logger.info("DeepFace rejected the uploaded image", exc_info=True)
        return AnalysisOutcome(message="The image could not be analysed reliably.")
    except Exception:
        logger.exception("DeepFace analysis failed")
        return AnalysisOutcome(
            message="Analysis could not be completed. Please try another image."
        )

    return interpret_result(raw_result, analysed_image, scale)


def _check_face_quality(
    result: Mapping[str, Any],
    analysed_image: np.ndarray,
    scale: float,
) -> str | None:
    try:
        detection_confidence = float(result.get("face_confidence", 0.0))
    except (TypeError, ValueError):
        detection_confidence = 0.0

    if detection_confidence < MIN_DETECTION_CONFIDENCE:
        return "The detected face is uncertain. Try a clearer, front-facing photo."

    region = result.get("region")
    if not isinstance(region, Mapping):
        return "The detected face could not be measured reliably."

    try:
        x = max(0, int(region["x"]))
        y = max(0, int(region["y"]))
        width = int(region["w"])
        height = int(region["h"])
    except (KeyError, TypeError, ValueError):
        return "The detected face could not be measured reliably."

    if width <= 0 or height <= 0:
        return "The detected face could not be measured reliably."
    if min(width, height) / scale < MIN_FACE_EDGE:
        return "The face is too far from the camera. Upload a closer portrait or crop the image."

    image_height, image_width = analysed_image.shape[:2]
    right = min(image_width, x + width)
    bottom = min(image_height, y + height)
    face = analysed_image[y:bottom, x:right]
    if face.size == 0:
        return "The detected face could not be measured reliably."

    grey_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    if float(cv2.Laplacian(grey_face, cv2.CV_64F).var()) < MIN_BLUR_SCORE:
        return "The face appears blurred. Use a sharper photo with less movement."

    brightness = float(grey_face.mean())
    if brightness < MIN_BRIGHTNESS:
        return "The face is too dark. Use a brighter, evenly lit photo."
    if brightness > MAX_BRIGHTNESS:
        return "The face is overexposed. Use a photo with softer, even lighting."
    if float(grey_face.std()) < MIN_CONTRAST:
        return "The facial details have too little contrast. Try a clearer, better-lit photo."

    return None
