"""
interpolate.py

This script reads a CSV containing tracked vehicle and license plate bounding boxes per frame,
interpolates missing frames' bounding boxes for each car, and writes a new CSV with the
interpolated data.
"""

# Importing Libraries
import csv
import numpy as np
from scipy.interpolate import interp1d
import os
import yaml

def interpolate_bounding_boxes(data):
    """
    Interpolate missing bounding boxes for each car across frames.

    Args:
        data (list of dict): List of dictionaries read from CSV containing keys:
                             'frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
                             'license_plate_bbox_score', 'license_number', 'license_number_score'.

    Returns:
        list of dict: New list of dictionaries with interpolated bounding boxes and frames filled in.
    """
    # Extract relevant columns
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    for car_id in unique_car_ids:
        # Get frame numbers for this car
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == car_id]

        # Mask for selecting this car's data
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]

        # Loop through frames for this car
        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                # Interpolate if frames are missing
                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)

                    interp_func_car = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interp_func_plate = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')

                    interpolated_car_bboxes = interp_func_car(x_new)
                    interpolated_license_plate_bboxes = interp_func_plate(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        # Build new rows
        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {
                'frame_nmr': str(frame_number),
                'car_id': str(car_id),
                'car_bbox': ' '.join(map(str, car_bboxes_interpolated[i])),
                'license_plate_bbox': ' '.join(map(str, license_plate_bboxes_interpolated[i]))
            }

            # If interpolated frame, set other values to '0'
            if str(frame_number) not in frame_numbers_:
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Keep original values if frame exists
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == car_id][0]
                row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
                row['license_number'] = original_row.get('license_number', '0')
                row['license_number_score'] = original_row.get('license_number_score', '0')

            interpolated_data.append(row)

    return interpolated_data

def main():
    """Main function to run interpolation and write CSV."""

    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Project root folder (parent of src)
    SRC_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
    RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Input CSV path (from result folder)
    input_csv = os.path.join(RESULT_DIR, cfg["raw_csv"])

    # Output CSV path in result folder
    csv_filename = cfg.get("interpolated_csv", "test_interpolated.csv")
    output_csv = os.path.join(RESULT_DIR, csv_filename)

    # Load CSV
    with open(input_csv, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Interpolate
    interpolated_data = interpolate_bounding_boxes(data)

    # Write new CSV
    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
    with open(output_csv, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)

    print(f"Interpolated CSV saved as {output_csv}")

if __name__ == "__main__":
    main()