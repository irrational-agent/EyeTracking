import os
import csv
import sqlite3
import base64
import shutil
from io import BytesIO
from PIL import Image
import argparse

def data_url_to_image(data_url: str):
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    return Image.open(BytesIO(data))

def setup_directories(base_dir, dataset_type):
    """Create directory structure for a dataset"""
    dataset_dir = os.path.join(base_dir, dataset_type)
    left_dir = os.path.join(dataset_dir, "left_eye")
    right_dir = os.path.join(dataset_dir, "right_eye")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    return dataset_dir, left_dir, right_dir

## TODO: Rewrite vibecoded functions

def export_gaze_dataset(cursor, output_dir, img_size=(128, 128)):
    """Export gaze dataset"""
    dataset_dir, left_dir, right_dir = setup_directories(output_dir, "gaze")
    csv_path = os.path.join(dataset_dir, "labels.csv")

    cursor.execute("""
        SELECT rowid, leftEyeFrame, rightEyeFrame, theta1, theta2
        FROM training_data
        WHERE leftEyeFrame != '' AND rightEyeFrame != '' AND type == 'gaze'
    """)
    rows = cursor.fetchall()

    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["left_image", "right_image", "theta1", "theta2"])
        for rowid, left_frame, right_frame, theta1, theta2 in rows:
            left_img = data_url_to_image(left_frame).convert("L").resize(img_size)
            right_img = data_url_to_image(right_frame).convert("L").resize(img_size)
            left_filename = f"left_{rowid}.png"
            right_filename = f"right_{rowid}.png"
            left_img.save(os.path.join(left_dir, left_filename))
            right_img.save(os.path.join(right_dir, right_filename))
            writer.writerow([
                os.path.join("left_eye", left_filename),
                os.path.join("right_eye", right_filename),
                theta1, theta2
            ])
    return len(rows)

def export_openness_dataset(cursor, output_dir, img_size=(128, 128)):
    """Export openness dataset"""
    dataset_dir, left_dir, right_dir = setup_directories(output_dir, "openness")
    csv_path = os.path.join(dataset_dir, "labels.csv")

    cursor.execute("""
        SELECT rowid, leftEyeFrame, rightEyeFrame, openness
        FROM training_data
        WHERE leftEyeFrame != '' AND rightEyeFrame != '' AND type == 'openness'
    """)
    rows = cursor.fetchall()

    with open(csv_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["left_image", "right_image", "openness"])
        for rowid, left_frame, right_frame, openness in rows:
            left_img = data_url_to_image(left_frame).convert("L").resize(img_size)
            right_img = data_url_to_image(right_frame).convert("L").resize(img_size)
            left_filename = f"left_{rowid}.png"
            right_filename = f"right_{rowid}.png"
            left_img.save(os.path.join(left_dir, left_filename))
            right_img.save(os.path.join(right_dir, right_filename))
            writer.writerow([
                os.path.join("left_eye", left_filename),
                os.path.join("right_eye", right_filename),
                openness
            ])
    return len(rows)

def export_dataset(db_path, output_dir, img_size=(128, 128), clear_folder=False):
    if clear_folder and os.path.exists(output_dir):
        print(f"Clearing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Exporting datasets from SQLite DB to folders with images and CSV labels.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    gaze_samples = export_gaze_dataset(cursor, output_dir, img_size)
    openness_samples = export_openness_dataset(cursor, output_dir, img_size)

    conn.close()
    print(f"Exported {gaze_samples} gaze samples to {os.path.join(output_dir, 'gaze')}")
    print(f"Exported {openness_samples} openness samples to {os.path.join(output_dir, 'openness')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export gaze and openness datasets from SQLite DB to folders with images and CSV labels.")
    parser.add_argument("--db_path", required=True, help="Path to the SQLite database file")
    parser.add_argument("--output_dir", required=False, help="Output directory for images and CSV", default="./dataset")
    parser.add_argument("--img_size", type=int, nargs=2, default=[128, 128], help="Image size as two integers (width height)")
    parser.add_argument("--clear", action="store_true", help="Clear output directory before exporting")
    args = parser.parse_args()

    export_dataset(args.db_path, args.output_dir, img_size=tuple(args.img_size), clear_folder=args.clear)
