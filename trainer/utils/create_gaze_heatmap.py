#!/usr/bin/env python
"""
Plot Theta Heatmap

Small utility tool for visualizing your training data.

This script connects to an SQLite database containing training data,
queries the columns theta1 and theta2 from the "training_data" table for rows
where both leftEyeFrame and rightEyeFrame are non-empty and the type is 'gaze',
and then plots a heatmap of these two values using a 2D histogram.
"""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def query_theta_values(db_path: str, limit: int = 50000):
    """
    Query theta1 and theta2 values from the SQLite database.

    Parameters:
        db_path (str): Path to the SQLite database.
        limit (int): Maximum number of rows to query.

    Returns:
        tuple: Two NumPy arrays containing theta1 and theta2 values.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = f"""
        SELECT theta1, theta2
        FROM training_data
        WHERE (leftEyeFrame != '' OR rightEyeFrame != '')
          AND type == 'gaze'
        ORDER BY RANDOM()
        LIMIT {limit}
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    # Unzip rows into two lists and convert to numpy arrays
    theta1, theta2 = zip(*rows) if rows else ([], [])
    return np.array(theta1), np.array(theta2)

def plot_heatmap(theta1: np.ndarray, theta2: np.ndarray, output_file: str = None):
    """
    Plot a heatmap of theta1 and theta2 using a 2D histogram.

    Parameters:
        theta1 (np.ndarray): Array of theta1 values.
        theta2 (np.ndarray): Array of theta2 values.
        output_file (str): If provided, save the plot to this file.
    """
    plt.figure(figsize=(8, 6))
    
    # Create a 2D histogram. Adjust bins as needed.
    hist = plt.hist2d(theta2, theta1, bins=50, cmap='viridis')
    plt.xlabel("Yaw")
    plt.ylabel("Pitch")
    plt.title("Heatmap of Pitch x Yaw")
    
    # Add a colorbar to show the count scale.
    plt.colorbar(hist[3], label='Counts')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_file}")
    else:
        plt.show()

def main():
    # Initialize Tkinter root and hide the main window.
    root = Tk()
    root.withdraw()

    # Use a system dialog to select the SQLite database file.
    db_path = filedialog.askopenfilename(
        title="Select SQLite Database File",
        filetypes=[("SQLite DB Files", "*.db"), ("All Files", "*.*")]
    )
    if not db_path:
        print("No database file selected. Exiting.")
        return

    # Optionally, ask the user for a file name to save the heatmap.
    output_file = filedialog.asksaveasfilename(
        title="Save Heatmap As",
        defaultextension=".png",
        filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
    )
    if output_file == "":
        output_file = None

    print("Querying theta values from the database...")
    theta1, theta2 = query_theta_values(db_path)
    print(f"Retrieved {len(theta1)} records.")
    
    print("Plotting heatmap...")
    plot_heatmap(theta1, theta2, output_file=output_file)

if __name__ == '__main__':
    main()
