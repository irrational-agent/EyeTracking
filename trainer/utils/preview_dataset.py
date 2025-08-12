
import sqlite3
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

def data_url_to_image(data_url: str):
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    return Image.open(BytesIO(data))

def preprocess_eye(data_url: str, size=(128, 128)):
    img = data_url_to_image(data_url).convert("L").resize(size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def get_random_samples(db_path, num_samples=8):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT leftEyeFrame, rightEyeFrame, theta1, theta2
        FROM training_data
        WHERE leftEyeFrame != '' AND rightEyeFrame != '' AND type == 'gaze'
        ORDER BY RANDOM()
        LIMIT {num_samples}
    """)
    rows = cursor.fetchall()
    conn.close()
    samples = []
    for left_frame, right_frame, theta1, theta2 in rows:
        left_img = preprocess_eye(left_frame, size=(128, 128))
        right_img = preprocess_eye(right_frame, size=(128, 128))
        combined_img = np.concatenate([left_img, right_img], axis=1).squeeze()
        samples.append((combined_img, (theta1, theta2)))
    return samples

def plot_samples(samples):
    n = len(samples)
    if n == 1:
        img, label = samples[0]
        plt.figure(figsize=(6, 3))
        plt.imshow(img, cmap='gray')
        plt.title(f"θ1: {label[0]:.2f}\nθ2: {label[1]:.2f}")
        plt.axis('off')
        plt.show(block=True)
    else:
        plt.figure(figsize=(12, 2.5 * ((n + 3) // 4)))
        for i, (img, label) in enumerate(samples):
            plt.subplot(1, n, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"θ1: {label[0]:.2f}\nθ2: {label[1]:.2f}")
            plt.axis('off')
        plt.tight_layout()
        plt.show(block=True)
    input("Press Enter to exit...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preview random samples from the eye tracking dataset.")
    parser.add_argument("--db_path", required=True, help="Path to the SQLite database file")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to preview")
    args = parser.parse_args()

    samples = get_random_samples(args.db_path, num_samples=args.num_samples)
    plot_samples(samples)