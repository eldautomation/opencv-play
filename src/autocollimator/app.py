from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np
from pathlib import Path
from typing import Any


from autocollimator.config.models import (
    Device,
    MeasurementOutput,
    MeasurementParameters,
    QualityLimits,
    ImageSensor,
    SourceOptics,
    DetectOptics,
    IOHardware,
    MeasurementHardware,
)

from autocollimator.config.store import (
    ConfigError,
    load_library,
    load_main,
    validate_device_references,
)

from autocollimator.workflow import (
    run_center_finding_on_image as workflow_run_center_finding_on_image,
)

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class AppContext:
    """
    Immutable application context.

    Parameters
    ----------
    config_dir:
        Directory containing main.toml and library/*.toml.
    output_dir:
        Directory where runtime outputs may be written.
    """

    config_dir: Path
    output_dir: Path
    input_dir: Path

class AutocollimatorApp:
    """
    First-pass application object for the autocollimator system.

    Responsibilities in this initial version:
    - Load configuration
    - Validate configuration references
    - Create output directory
    - Expose current configuration to calling code

    Future versions can extend this class to:
    - Initialize hardware
    - Capture images
    - Run target-finding workflows
    - Generate reports
    """

    def __init__(
        self,
        config_dir: str | Path = "configs",
        output_dir: str | Path = "outputs",
        input_dir: str | Path = "inputs",        
    ) -> None:
        self.context = AppContext(
            config_dir=Path(config_dir).resolve(),
            output_dir=Path(output_dir).resolve(),
            input_dir=Path(input_dir).resolve(),
        )

        self.library: dict[str, Any] | None = None
        self.device_config: Device | None = None
        self.measurement_parameters: MeasurementParameters | None = None
        self.is_started: bool = False

        LOGGER.debug(
            "AutocollimatorApp initialized with config_dir=%s input_dir=%s output_dir=%s",
            self.context.config_dir,
            self.context.input_dir,
            self.context.output_dir,
        )

    def startup(self) -> None:
        """
        Start the application.

        This loads configuration files, validates cross-references,
        and prepares the output directory.

        Raises
        ------
        FileNotFoundError
            If required config files are missing.
        ConfigError
            If configuration is invalid.
        RuntimeError
            If the application has already been started.
        """
        if self.is_started:
            raise RuntimeError("Application is already started")

        LOGGER.info("Starting application")
        LOGGER.debug("Loading library from %s", self.context.config_dir)

        self.library = load_library(self.context.config_dir)

        LOGGER.debug("Loading main config from %s", self.context.config_dir)
        self.device_config = load_main(self.context.config_dir)

        LOGGER.debug("Validating device references")
        validate_device_references(self.device_config, self.library)

        self.context.output_dir.mkdir(parents=True, exist_ok=True)

        self.is_started = True
        LOGGER.info("Application startup complete")

    def shutdown(self) -> None:
        """
        Shut down the application.

        In this initial version, there are no persistent hardware or external
        resources to release. This method exists to establish the lifecycle API.

        Safe to call more than once.
        """
        if not self.is_started:
            LOGGER.debug("Shutdown called while application was not started")
            return

        LOGGER.info("Shutting down application")

        self.library = None
        self.device_config = None
        self.is_started = False

        LOGGER.info("Application shutdown complete")

    def get_current_config(self) -> Device:
        """
        Return the currently loaded device configuration.

        Returns
        -------
        Device
            The active device configuration loaded from main.toml.

        Raises
        ------
        RuntimeError
            If startup() has not been called yet.
        """
        if not self.is_started or self.device_config is None:
            raise RuntimeError("Application is not started. Call startup() first.")

        return self.device_config

    def get_library(self) -> dict[str, Any]:
        """
        Return the currently loaded configuration library.

        Returns
        -------
        dict[str, Any]
            Library data loaded from configs/library/*.toml.

        Raises
        ------
        RuntimeError
            If startup() has not been called yet.
        """
        if not self.is_started or self.library is None:
            raise RuntimeError("Application is not started. Call startup() first.")

        return self.library

    def validate(self) -> None:
        """
        Re-validate the current configuration state.

        Raises
        ------
        RuntimeError
            If startup() has not been called yet.
        ConfigError
            If the loaded configuration is invalid.
        """
        if not self.is_started or self.library is None or self.device_config is None:
            raise RuntimeError("Application is not started. Call startup() first.")

        validate_device_references(self.device_config, self.library)
        LOGGER.debug("Configuration re-validation passed")

    def status(self) -> dict[str, Any]:
        """
        Return a simple application status snapshot.

        Returns
        -------
        dict[str, Any]
            Basic state information useful for debugging or UI display.
        """
        return {
            "is_started": self.is_started,
            "config_dir": str(self.context.config_dir),
            "output_dir": str(self.context.output_dir),
            "has_library": self.library is not None,
            "has_device_config": self.device_config is not None,
        }

    def __enter__(self) -> "AutocollimatorApp":
        """
        Context-manager entry.

        Starts the application automatically.
        """
        self.startup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Context-manager exit.

        Shuts the application down automatically.
        """
        self.shutdown()

    # helper functions for software settings
    def get_measurement_parameters(self) -> MeasurementParameters:
        if not self.is_started:
            raise RuntimeError("Application is not started. Call startup() first.")

        device = self.get_current_config()
        lib = self.get_library()

        return lib["measurement_parameters_by_id"][device.measurement_parameters_id]

    def get_quality_limits(self) -> QualityLimits:
        if not self.is_started:
            raise RuntimeError("Application not started.")

        device = self.get_current_config()
        lib = self.get_library()

        return lib["quality_limits_by_id"][device.quality_limits_id]

    def get_image_sensor(self) -> ImageSensor:
        """
        Return the active image sensor configuration referenced by the current device.
        """
        if not self.is_started:
            raise RuntimeError("Application is not started. Call startup() first.")

        device = self.get_current_config()
        lib = self.get_library()

        return lib["sensors_by_id"][device.image_sensor_id]


    def get_source_optics(self) -> SourceOptics:
        """
        Return the active source optics configuration referenced by the current device.
        """
        if not self.is_started:
            raise RuntimeError("Application is not started. Call startup() first.")

        device = self.get_current_config()
        lib = self.get_library()

        return lib["source_by_id"][device.source_optics_id]


    def get_detect_optics(self) -> DetectOptics:
        """
        Return the active detect optics configuration referenced by the current device.
        """
        if not self.is_started:
            raise RuntimeError("Application is not started. Call startup() first.")

        device = self.get_current_config()
        lib = self.get_library()

        return lib["detect_by_id"][device.detect_optics_id]


    def get_io_hardware(self) -> IOHardware:
        """
        Return the active IO hardware configuration referenced by the current device.
        """
        if not self.is_started:
            raise RuntimeError("Application is not started. Call startup() first.")

        device = self.get_current_config()
        lib = self.get_library()

        return lib["io_by_id"][device.io_hardware_id]


    def get_measurement_hardware(self) -> MeasurementHardware:
        """
        Return a fully resolved MeasurementHardware object for the active device.
        """
        if not self.is_started:
            raise RuntimeError("Application is not started. Call startup() first.")

        return MeasurementHardware(
            device=self.get_current_config(),
            image_sensor=self.get_image_sensor(),
            source_optics=self.get_source_optics(),
            detect_optics=self.get_detect_optics(),
            io_hardware=self.get_io_hardware(),
        )


    # Here begin the funcitonal elements
 
    # What do we pass to the image parsing?
    # What image parse variables are persistent?
    def run_center_finding_on_image(
        self,
        image: np.ndarray,
        *,
        debug: bool = False,
        debug_prefix: str | Path | None = None,
    ) -> MeasurementOutput:
        """
        User-facing wrapper for the center-finding workflow.
        """
        if not self.is_started:
            raise RuntimeError("Application is not started. Call startup() first.")

        # mps = self.get_measurement_parameters()
    
        # print(f"measurement parameters are: \n{mps}")

        return workflow_run_center_finding_on_image(
            app=self,
            image=image,
            debug=debug,
            debug_prefix=debug_prefix,
        )
    
    def save_measurement_output(self,measurement_result):
        print("save_measurement_output not yet implemented")
        pass













