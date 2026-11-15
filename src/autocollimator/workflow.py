from __future__ import annotations

import logging
import math
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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
    PresearchRecord,
    PresearchResult,
)

from autocollimator.instrument import InstrumentModel, check_image_size, to_angular
from autocollimator.presearch import presearch, recommend_region
from autocollimator.target_utils import(
    ROI_NAMES,
    CrossCenterResult,
    draw_box_on_image,
    ensure_bgr_u8,
    ensure_gray_f64,
    find_cross_center_detailed,
    image_md5
)
from autocollimator.units import OutputUnits

if TYPE_CHECKING:
    from autocollimator.app import AutocollimatorApp


LOGGER = logging.getLogger(__name__)

# Overlay color (BGR) of the pre-search's recommended region: orange, distinct from the debug
# overlay's blue crop, yellow ROIs, green ROI centers, red center and magenta lines.
PRESEARCH_REGION_COLOR = (0, 165, 255)

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


def _quality_failure_message(q_ratio_list: list[float], q_limit: float) -> str:
    """
    Describe which ROIs failed the RSS quality check (same test as sdrm_2: ratio not finite
    or ratio > limit).
    """
    not_finite = [
        f"ROI {name}: quality ratio not finite (flat profile)"
        for name, q in zip(ROI_NAMES, q_ratio_list)
        if not math.isfinite(q)
    ]
    failed = [
        f"{name} (rss_ratio {q:.3f} > limit {q_limit:.3f})"
        for name, q in zip(ROI_NAMES, q_ratio_list)
        if math.isfinite(q) and q > q_limit
    ]
    if not_finite or failed:
        parts = []
        if failed:
            parts.append("quality check failed for ROI " + ", ".join(failed))
        parts.extend(not_finite)
        return "Center could not be determined: " + "; ".join(parts)
    ratios = ", ".join(f"{name} {q:.3f}" for name, q in zip(ROI_NAMES, q_ratio_list))
    return f"Center could not be determined (rss_ratio limit {q_limit:.3f}; ratios: {ratios})"


@dataclass(frozen=True)
class _CropChoice:
    """Which crop a measurement used and how it was chosen (see presearch_mode)."""

    # None when nothing was measured (pre-search failed, presearch_failure_action "fail").
    result: CrossCenterResult | None
    crop_center: tuple[int, int] | None
    crop_size: tuple[int, int] | None
    # "configured" | "presearch" | "tracked"; None when nothing was measured.
    crop_source: str | None
    presearch_ran: bool
    presearch_result: PresearchResult | None
    # Appended to the measurement message; the whole message when nothing was measured.
    note: str | None


def _choose_crop_and_measure(
    app: "AutocollimatorApp",
    image: np.ndarray,
    model: InstrumentModel,
    mps: MeasurementParameters,
    measure: Callable[[tuple[int, int], tuple[int, int]], CrossCenterResult],
    configured_center: tuple[int, int],
    configured_size: tuple[int, int],
) -> _CropChoice:
    """
    Pick the crop by ``mps.presearch.presearch_mode`` and measure with it.

    - "off": the configured crop; the pre-search is not run.
    - "every_image": pre-search; on success measure with the recommended crop.
    - "track": measure around the previous successful center (app memory) with the
      recommended size; if there is none, or that measurement fails, pre-search as above.
    A failed pre-search applies ``presearch_failure_action``: "use_configured_crop" measures
    with the configured crop and notes the failure, "fail" measures nothing.
    """
    ps = mps.presearch
    if ps.presearch_mode == "off":
        result = measure(configured_center, configured_size)
        return _CropChoice(result, configured_center, configured_size, "configured", False, None, None)

    height, width = image.shape[:2]
    if ps.presearch_mode == "track" and app.tracked_center is not None:
        region = recommend_region(app.tracked_center, (width, height), ps, app.tracked_line_fwhm_px)
        if region.center is not None and region.size is not None:
            size = (region.size, region.size)
            result = measure(region.center, size)
            if result.center is not None:
                return _CropChoice(result, region.center, size, "tracked", False, None, None)
            LOGGER.info("Measurement at the tracked center %s failed; running the pre-search", app.tracked_center)
        else:
            LOGGER.info(
                "No region at the tracked center %s (%s); running the pre-search",
                app.tracked_center, region.failure_reason,
            )

    pr = presearch(image, ps, model.reticle_type, gray_mode=mps.gray_mode, roi_depth_px=int(mps.roi_size_y))
    if pr.success:
        assert pr.recommended_center_x is not None and pr.recommended_center_y is not None
        assert pr.recommended_size is not None
        center = (pr.recommended_center_x, pr.recommended_center_y)
        size = (pr.recommended_size, pr.recommended_size)
        return _CropChoice(measure(center, size), center, size, "presearch", True, pr, None)

    LOGGER.warning("Pre-search failed: %s (presearch_failure_action %s)", pr.failure_reason, ps.presearch_failure_action)
    if ps.presearch_failure_action == "fail":
        return _CropChoice(None, None, None, None, True, pr, f"Pre-search failed: {pr.failure_reason}")
    result = measure(configured_center, configured_size)
    return _CropChoice(
        result, configured_center, configured_size, "configured", True, pr,
        f"pre-search failed: {pr.failure_reason}; configured crop used",
    )


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
    units: OutputUnits | None = None,
) -> tuple[MeasurementOutput, np.ndarray]:
    """
    Run the center-finding workflow on an input image.

    Composes image -> image measurement (MeasuredValues, pixels) -> angular measurement
    (instrument.to_angular) -> MeasurementOutput with an InstrumentModel snapshot.

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
    units:
        Output units for the angular measurement; None uses the device defaults.

    Returns
    -------
    tuple[MeasurementOutput, np.ndarray]
        Structured workflow result and the overlay image.

    Raises
    ------
    RuntimeError
        If the app is not started.
    TypeError, ValueError
        If inputs are invalid.
    ImageSizeMismatchError
        If the image size differs from the capture mode's output size and the device's
        image_size_mismatch policy is "error".
    """


    if not app.is_started:
        raise RuntimeError("Application is not started. Call startup() first.")

    image = _validate_image(image)
    image_height, image_width = image.shape[:2]

    model = app.get_instrument_model()
    if units is None:
        units = app.get_output_units()
    check_image_size(model, image_width, image_height)

    # pull values out of measurement parameters, for validation


    mps = app.get_measurement_parameters()
    qls = app.get_quality_limits()

    crop_center = (int(mps.crop_center_x),int(mps.crop_center_y))
    crop_size = (int(mps.crop_size_x),int(mps.crop_size_y))
    # roi_size_x is an int or "auto"; MeasurementParameters and find_cross_center validate it.
    roi_size = (mps.roi_size_x, int(mps.roi_size_y))

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
        "Running center-finding workflow with crop_center=%s crop_size=%s roi_size=%s "
        "subpixel=%s subpixel_neighbors=%s profile_mode=%s oversample_step=%s fine_window=%s "
        "fine_window_mode=%s refine_search=%s max_refine_shift=%s gray_mode=%s debug=%s",
        crop_center,
        crop_size,
        roi_size,
        mps.subpixel,
        mps.subpixel_neighbors,
        mps.profile_mode,
        mps.oversample_step,
        mps.fine_window,
        mps.fine_window_mode,
        mps.refine_search,
        mps.max_refine_shift,
        mps.gray_mode,
        debug,
    )

    def measure(center: tuple[int, int], size: tuple[int, int]) -> CrossCenterResult:
        return find_cross_center_detailed(
            image=image,
            crop_center=center,
            crop_size=size,
            roi_size=roi_size,
            q_limit=qls.rss_ratio,
            nominal_slant=mps.nominal_slant,
            subpixel=mps.subpixel,
            subpixel_neighbors=mps.subpixel_neighbors,
            profile_mode=mps.profile_mode,
            oversample_step=mps.oversample_step,
            fine_window=mps.fine_window,
            fine_window_mode=mps.fine_window_mode,
            refine_search=mps.refine_search,
            max_refine_shift=mps.max_refine_shift,
            gray_mode=mps.gray_mode,
            debug=debug,
            debug_prefix=debug_prefix_str if debug_prefix_str is not None else "",
        )

    choice = _choose_crop_and_measure(app, image, model, mps, measure, crop_center, crop_size)
    result = choice.result
    if result is not None:
        position, angles = result.center, result.angles
        overlay_image, q_ratio_list = result.overlay, result.q_ratio_list
        roi_methods, saturation, fine_windows = result.roi_methods, result.saturation, result.fine_windows
        success = position is not None
        message = "Center found" if success else _quality_failure_message(q_ratio_list, qls.rss_ratio)
        if choice.note is not None:
            message = f"{message} ({choice.note})"
    else:
        # Pre-search failed with presearch_failure_action "fail": center finding did not run,
        # so there are no ROI values (NaN quality ratios, "not_run" methods).
        position, angles = None, None
        overlay_image = ensure_bgr_u8(ensure_gray_f64(image, mps.gray_mode))
        q_ratio_list = [math.nan] * len(ROI_NAMES)
        roi_methods = ("not_run",) * len(ROI_NAMES)
        saturation = fine_windows = (None,) * len(ROI_NAMES)
        success = False
        message = choice.note or "Pre-search failed"
    if not success:
        LOGGER.warning(message)

    LOGGER.info(
        "Center-finding result success=%s position=%s angles=%s",
        success,
        position,
        angles,
    )
    if success:
        app.set_tracking(
            position,
            choice.presearch_result.line_fwhm_px if choice.presearch_result is not None else app.tracked_line_fwhm_px,
        )

    presearch_result = choice.presearch_result
    if presearch_result is not None and presearch_result.success:
        # Recommended region, in a color of its own (PRESEARCH_REGION_COLOR).
        assert presearch_result.recommended_size is not None
        size = presearch_result.recommended_size
        x0 = presearch_result.recommended_center_x - size // 2
        y0 = presearch_result.recommended_center_y - size // 2
        overlay_image = draw_box_on_image(
            overlay_image, (x0, y0), (x0 + size - 1, y0 + size - 1), width=2, color=PRESEARCH_REGION_COLOR
        )

    overlay_hash = image_md5(overlay_image)
    image_hash = image_md5(image)

    measured_values = MeasuredValues(
        center_position_x = position[0] if success else None,
        center_position_y = position[1] if success else None,
        measured_angle_a0 = angles[0] if success else None,
        measured_angle_a1 = angles[1] if success else None,
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
        message = message,
        roi_method_m0 = roi_methods[0],
        roi_method_m1 = roi_methods[1],
        roi_method_m2 = roi_methods[2],
        roi_method_m3 = roi_methods[3],
        saturation_s0 = saturation[0],
        saturation_s1 = saturation[1],
        saturation_s2 = saturation[2],
        saturation_s3 = saturation[3],
        fine_window_used_f0 = fine_windows[0],
        fine_window_used_f1 = fine_windows[1],
        fine_window_used_f2 = fine_windows[2],
        fine_window_used_f3 = fine_windows[3],
        image_width = image_width,
        image_height = image_height,
    )
    angular_measurement = to_angular(measured_values, model, units)
    LOGGER.info(
        "Angular result success=%s tilt=(%s, %s) %s rotation=%s orthogonality=%s %s",
        angular_measurement.success,
        angular_measurement.tilt_x,
        angular_measurement.tilt_y,
        units.tilt_unit.value,
        angular_measurement.rotation,
        angular_measurement.orthogonality,
        units.rotation_unit.value,
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
        measurement_name = "Measurement_Placeholder",
        image_name = "Image_Placeholder",
        image_hash = image_hash,
        angular_measurement = angular_measurement,
        instrument_model = model,
        presearch = PresearchRecord(
            mode = mps.presearch.presearch_mode,
            ran = choice.presearch_ran,
            result = choice.presearch_result,
            crop_center_x = None if choice.crop_center is None else choice.crop_center[0],
            crop_center_y = None if choice.crop_center is None else choice.crop_center[1],
            crop_size_x = None if choice.crop_size is None else choice.crop_size[0],
            crop_size_y = None if choice.crop_size is None else choice.crop_size[1],
            crop_source = choice.crop_source,
        ),
    )

    return measurement_output, overlay_image

