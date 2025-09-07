"""
visualize.py

This script reads interpolated vehicle tracking data from a CSV and overlays bounding boxes,
license plate crops, and plate numbers on a video. The output video is saved with all annotations.
"""

# Importing Libraries
import ast
import cv2
import numpy as np
import pandas as pd
import os
import yaml


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """
    Draws stylized corner borders around a rectangle (for cars).

    Args:
        img (np.ndarray): The image on which to draw.
        top_left (tuple): Top-left coordinates (x, y).
        bottom_right (tuple): Bottom-right coordinates (x, y).
        color (tuple): Color in BGR format.
        thickness (int): Line thickness.
        line_length_x (int): Length of horizontal corner lines.
        line_length_y (int): Length of vertical corner lines.

    Returns:
        np.ndarray: Image with drawn borders.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left corner
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)

    return img


def main():
    # Load config
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup project root and result folder
    SRC_DIR = os.path.dirname(__file__)
    PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
    RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Load the interpolated tracking results
    interpolated_path = os.path.join(RESULT_DIR, cfg["interpolated_csv"])
    results = pd.read_csv(interpolated_path)
    
    # Load the video
    video_path = os.path.abspath(cfg["input_video"])
    cap = cv2.VideoCapture(video_path)

    # Setup output video path in result folder
    output_filename = os.path.basename(cfg.get("output_video", "out.mp4"))
    output_path = os.path.join(RESULT_DIR, output_filename)

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Preprocess license plate crops for each car
    license_plate = {}
    for car_id in np.unique(results['car_id']):
        # Get the frame with the highest confidence license plate
        max_score = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        best_row = results[(results['car_id'] == car_id) &
                           (results['license_number_score'] == max_score)].iloc[0]
        
        # Set license plate number and frame crop
        license_plate[car_id] = {'license_crop': None,
                                 'license_plate_number': best_row['license_number']}
        
        # Move to the frame containing the license plate
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_row['frame_nmr']))
        ret, frame = cap.read()

        # Extract bounding box
        x1, y1, x2, y2 = ast.literal_eval(best_row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        # Resize crop for display (height 400)
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
        license_plate[car_id]['license_crop'] = license_crop

    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Process frames
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if not ret:
            break

        # Filter results for current frame
        df_ = results[results['frame_nmr'] == frame_nmr]

        for row_indx in range(len(df_)):
            row = df_.iloc[row_indx]

            # Draw car bounding box with corner borders
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(row['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # Draw license plate rectangle
            x1, y1, x2, y2 = ast.literal_eval(row['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Overlay license plate crop above car
            license_crop = license_plate[row['car_id']]['license_crop']
            H, W, _ = license_crop.shape

            try:
                # Place crop above car
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                # White background for text
                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                # Overlay license plate number
                text = license_plate[row['car_id']]['license_plate_number']
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
                cv2.putText(frame, text, (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)

            except Exception:
                # Skip if overlay goes outside frame
                pass

        # Write annotated frame
        out.write(frame)

    # Release resources
    out.release()
    cap.release()

if __name__ == "__main__":
    main()