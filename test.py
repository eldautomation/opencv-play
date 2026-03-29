from pathlib import Path
from autocollimator.config.models import ImageSensor
from autocollimator.config.store import (
    add_image_sensor,
    load_library,
    load_main,
    validate_device_references,
)

CONFIG_DIR = Path("configs")

lib = load_library(CONFIG_DIR)
device = load_main(CONFIG_DIR)
validate_device_references(device, lib)

add_image_sensor(
    CONFIG_DIR,
    ImageSensor(
        id=3,
        name="IMX900",
        manufacturer="Sony",
        part_number="IMX900",
        pixel_size_x=3.2,
        pixel_size_y=3.2,
        pixels_x=2048,
        pixels_y=2048,
    ),
)
