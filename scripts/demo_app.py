from pathlib import Path
import logging
import cv2

from autocollimator.app import AutocollimatorApp
from autocollimator.target_utils import pct_list_to_int_list


# note: use run with this: PYTHONPATH=src python scripts/demo_app.py



def main() -> int:
    # logging.basicConfig(level=logging.INFO)

    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True,exist_ok=True)

    # 4 levels - INFO, WARNING, ERROR, DEBUG
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
        logging.FileHandler(log_dir / "app.log"),
        logging.StreamHandler(),
        ],
    )
 
    logging.getLogger("autocollimator").setLevel(logging.DEBUG)
    
    # showing design pattern only. 
    # logging.getLogger("autocollimator.target_utils").setLevel(logging.DEBUG)
    # logging.getLogger("autocollimator.workflow").setLevel(logging.DEBUG)


    config_dir = Path("configs")
    output_dir = Path("outputs/demo")
    debug_dir = Path("outputs/demo/debug")
    source_dir = Path("tests/assets/images")
    log_dir = Path("outputs/logs")

    log_dir.mkdir(parents=True,exist_ok=True)
    debug_dir.mkdir(parents=True,exist_ok=True)
    output_dir.mkdir(parents=True,exist_ok=True)

    app = AutocollimatorApp(config_dir=config_dir, input_dir=source_dir, output_dir=output_dir)

    try:
        app.startup()

        print("\n--- App Status ---")
        print(app.status())

        print("\n--- Current Config ---")
        cfg = app.get_current_config()
        print(cfg)

        print("\n--- Config Validation ---")
        app.validate()
        print("Config is valid.")


        image_names = [
            "blob-05.jpg",
            "blob-06.jpg",
            "blob-08.jpg",
            "blob-09.jpg",
            "blob-11.jpg",
            "blob-12.jpg",
        ]

        

        cx_list = [0.47, 0.6, 0.27, 0.6, 0.3, 0.6, 0.5]
        cy_list = [0.5, 0.7, 0.35, 0.5, 0.4, 0.4, 0.4]
        x_pct_list = [0.6, 0.6, 0.4, 0.7, 0.4, 0.6, 0.6]
        y_pct_list = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]


        device = app.get_current_config()
        lib = app.get_library()
        sensor = lib["sensors_by_id"][device.image_sensor_id]
        width = sensor.pixels_x
        height = sensor.pixels_y

        cx_px = pct_list_to_int_list(cx_list, width)
        cy_px = pct_list_to_int_list(cy_list, height)
        x_crop_px = pct_list_to_int_list(x_pct_list, width)
        y_crop_px = pct_list_to_int_list(y_pct_list, height)

        ###


        ###
        # def run_center_finding_on_image(
        #     app: "AutocollimatorApp",
        #     image: np.ndarray,
        #     *,
        #     crop_center: tuple[int, int] | None = None,
        #     crop_size: tuple[int, int] | None = None,
        #     roi_size: tuple[int, int] = (500, 20),
        #     debug: bool = False,
        #     debug_prefix: str | Path | None = None,
        ###

        for i in range(len(image_names)):
            cx0 = cx_px[i]
            cy0 = cy_px[i]
            x_crop0 = x_crop_px[i]
            y_crop0 = y_crop_px[i]
            image_name = image_names[i]

            print(f"i is:{i}\t\t name is:{image_name}")            
            if "09" not in image_name:
                print(f"Rejecting image - looking for '09' ")
                continue


            print("\n--- Image Processing ---")
            #
            # Do a test of the image "blob-05.jpg"
            #
            image_path = Path(source_dir / image_name)
            source_image = cv2.imread(str(image_path))
            if source_image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")


            print(f"cy,cy:{cx0},{cy0}\t\tx_crop,y_crop:{x_crop0},{y_crop0}\t\troi_size:500,20")

            debug_name = "debug-"+image_name[:-4]

            measurement_output, overlay_image = app.run_center_finding_on_image(
                image=source_image,
                debug=True,
                debug_prefix=Path(debug_dir / debug_name),
            )

            print(f"Result - success:{measurement_output.measured_values.success}")

            measurement_output = measurement_output

            app.save_measurement_output(measurement_output,source_image,overlay_image)

            # print(result)


        return 0

    finally:
        app.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

