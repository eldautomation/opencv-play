from __future__ import annotations

import logging
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autocollimator.config.models import (
    Device,
    DetectOptics,
    ImageSensor,
    IOHardware,
    SourceOptics,
    MeasurementParameters,
    QualityLimits,
    MeasuredValues,
    Recommendations,
    MeasurementHardware,
    MeasurementOutput,
)

from autocollimator.target_utils import(
    find_cross_center,
    image_md5
) 

if TYPE_CHECKING:
    from autocollimator.app import AutocollimatorApp


LOGGER = logging.getLogger(__name__)

# @dataclass(frozen=True)
# class CenterFindingResult:
#     """
#     Result of a center-finding workflow run.
#     """

#     success: bool
#     position: tuple[float, float] | None
#     angles: tuple[float, float] | None
#     crop_center: tuple[int, int] | None
#     crop_size: tuple[int, int] | None
#     roi_size: tuple[int, int]
#     debug_prefix: str | None
#     message: str


def _validate_image(image: np.ndarray) -> np.ndarray:
    """
    Validate and normalize image input.

    Parameters
    ----------
    image:
        Input image as a NumPy array.

    Returns
    -------
    np.ndarray
        Validated image array.

    Raises
    ------
    TypeError
        If image is not a NumPy array.
    ValueError
        If image is empty or has invalid dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("image is empty")

    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}")

    return image


def _default_crop_from_image(image: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Compute a simple default crop region from the image.

    Current behavior:
    - crop center = center of image
    - crop size = half the image width/height

    This is a reasonable first-pass fallback until config-driven crop logic is added.
    """
    height, width = image.shape[:2]

    crop_center = (width // 2, height // 2)
    crop_size = (max(1, width // 1.3), max(1, height // 1.3))
    crop_size = ( int(crop_size[0]) , int(crop_size[1]) )

    return crop_center, crop_size


            # app=self,
            # image=image,
            # mps=mps,
            # debug=debug,
            # debug_prefix=debug_prefix,

def run_center_finding_on_image(
    app: "AutocollimatorApp",
    image: np.ndarray,
    *,
    debug: bool = False,
    debug_prefix: str | Path | None = None,
) -> MeasurementOutput:
    """
    Run the center-finding workflow on an input image.

    Parameters
    ----------
    app:
        Running application instance.
    image:
        Input image as a NumPy array.
    crop_center:
        Optional crop center in pixels. If omitted, a default is computed.
    crop_size:
        Optional crop size in pixels. If omitted, a default is computed.
    roi_size:
        ROI size passed through to ``find_cross_center``.
    debug:
        Whether debug outputs should be generated.
    debug_prefix:
        Optional file prefix for debug artifacts.

    Returns
    -------
    MeasurementOutput
        Structured workflow result.

    Raises
    ------
    RuntimeError
        If the app is not started.
    TypeError, ValueError
        If inputs are invalid.
    """


    if not app.is_started:
        raise RuntimeError("Application is not started. Call startup() first.")

    image = _validate_image(image)

    # pull values out of measurement parameters, for validation


    mps = app.get_measurement_parameters()
    qls = app.get_quality_limits()

    crop_center = (int(mps.crop_center_x),int(mps.crop_center_y))
    crop_size = (int(mps.crop_size_x),int(mps.crop_size_y))
    roi_size = (int(mps.roi_size_x),int(mps.roi_size_y))


    if (
        not isinstance(roi_size, tuple)
        or len(roi_size) != 2
        or not all(isinstance(v, int) for v in roi_size)
    ):
        raise TypeError("roi_size must be a tuple[int, int]")

    if roi_size[0] <= 0 or roi_size[1] <= 0:
        raise ValueError(f"roi_size must contain positive integers, got {roi_size}")

    if not isinstance(debug, bool):
        raise TypeError(f"debug must be bool, got {type(debug).__name__}")

    if debug_prefix is None and debug:
        debug_prefix = app.context.output_dir / "centerfinding"

    debug_prefix_str: str | None
    if debug_prefix is None:
        debug_prefix_str = None
    else:
        debug_prefix_path = Path(debug_prefix)
        debug_prefix_path.parent.mkdir(parents=True, exist_ok=True)
        debug_prefix_str = str(debug_prefix_path)

    if crop_center is None or crop_size is None:
        default_center, default_size = _default_crop_from_image(image)
        if crop_center is None:
            crop_center = default_center
        if crop_size is None:
            crop_size = default_size

    if (
        not isinstance(crop_center, tuple)
        or len(crop_center) != 2
        or not all(isinstance(v, int) for v in crop_center)
    ):
        raise TypeError("crop_center must be a tuple[int, int]")

    if (
        not isinstance(crop_size, tuple)
        or len(crop_size) != 2
        or not all(isinstance(v, int) for v in crop_size)
    ):
        LOGGER.info(f"crop size is:{crop_size}")
        raise TypeError("crop_size must be a tuple[int, int]")

    if crop_size[0] <= 0 or crop_size[1] <= 0:
        raise ValueError(f"crop_size must contain positive integers, got {crop_size}")

    LOGGER.info(
        "Running center-finding workflow with crop_center=%s crop_size=%s roi_size=%s debug=%s",
        crop_center,
        crop_size,
        roi_size,
        debug,
    )

    position, angles, overlay_image, q_ratio_list = find_cross_center(
        image=image,
        crop_center=crop_center,
        crop_size=crop_size,
        roi_size=roi_size,
        q_limit=qls.rss_ratio,
        debug=debug,
        debug_prefix=debug_prefix_str if debug_prefix_str is not None else "",
    )

    success = position is not None
    message = "Center found" if success else "Center could not be determined"

    LOGGER.info(
        "Center-finding result success=%s position=%s angles=%s",
        success,
        position,
        angles,
    )
    # LOGGER.debug(f"q ratio:{q_ratio_list}")

    overlay_hash = image_md5(overlay_image)
    image_hash = image_md5(image)

    measured_values = MeasuredValues(
        center_position_x = position[0],
        center_position_y = position[1],
        measured_angle_a0 = angles[0],
        measured_angle_a1 = angles[1],
        rss_ratio_r0 = q_ratio_list[0],
        rss_ratio_r1 = q_ratio_list[1],
        rss_ratio_r2 = q_ratio_list[2],
        rss_ratio_r3 = q_ratio_list[3],
        brightness_ratio_b0 = 0,
        brightness_ratio_b1 = 0,
        brightness_ratio_b2 = 0,
        brightness_ratio_b3 = 0,
        linewidth_w0 = 0,
        linewidth_w1 = 0,
        linewidth_w2 = 0,
        linewidth_w3 = 0,
        overlay_hash = overlay_hash,
        success = success,
        message = message   
    )
    recommendations = Recommendations(
        updated_brightness=0,
        updated_crop_center_x = 0,
        updated_crop_center_y = 0,
        updated_crop_size_x=0,
        updated_crop_size_y=0,
        updated_roi_size_x=0,
        updated_roi_size_y=0,
    )

    measurement_output = MeasurementOutput(
        hardware = app.get_measurement_hardware(),
        measurement_parameters = mps,
        quality_limits = qls,
        measured_values = measured_values,
        recommendations = recommendations,
        image_name = "Test",
        image_hash = image_hash,
        output_units = "Pixels",
        pixel_to_unit_scale_factor = 0,
    )

    return measurement_output, overlay_image

