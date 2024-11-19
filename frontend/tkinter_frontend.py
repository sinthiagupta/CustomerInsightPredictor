import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import time
import logging
import csv
import os

# Path setup for modular imports
import sys
sys.path.append("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor")

from src.segmentation_model import perform_segmentation
from src.prediction_model import perform_prediction

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Metrics File
METRICS_FILE = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/model_metrics.csv"

# Create the metrics file if it doesn't exist
if not os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Operation", "Execution Time (s)", "Timestamp"])

# Function to browse file
def browse_file(filetypes, initial_dir, title):
    return filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=initial_dir)

# Function to display messages
def display_message(title, message):
    messagebox.showinfo(title, message)

def display_error(title, message):
    messagebox.showerror(title, message)

# Function to save metrics
def save_metrics(operation, execution_time):
    with open(METRICS_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([operation, f"{execution_time:.2f}", time.strftime("%Y-%m-%d %H:%M:%S")])

# Segmentation Functionality
def segmentation():
    file_path = browse_file([("CSV Files", "*.csv")], "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/data/", "Select Dataset for Segmentation")
    
    if file_path:
        output_path = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/segmentation_results.csv"
        
        start_time = time.time()
        perform_segmentation(file_path, output_path)
        end_time = time.time()
        
        execution_time = end_time - start_time
        save_metrics("Segmentation", execution_time)
        display_message("Segmentation Complete", f"Results saved to {output_path}\nTime taken: {execution_time:.2f} seconds")

        # Display results in GUI
        display_results(output_path, "Segmentation Results")

# Prediction Functionality
def prediction():
    file_path = browse_file([("CSV Files", "*.csv")], "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/data/", "Select Dataset for Prediction")
    
    if file_path:
        output_path = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/prediction_result.csv"
        
        try:
            # Pass configuration for prediction
            config = {"file_path": file_path, "output_path": output_path}
            
            start_time = time.time()
            perform_prediction(config)
            end_time = time.time()
            
            execution_time = end_time - start_time
            save_metrics("Prediction", execution_time)
            display_message("Prediction Complete", f"Results saved to {output_path}\nTime taken: {execution_time:.2f} seconds")

            # Display results in GUI
            display_results(output_path, "Prediction Results")
        except Exception as e:
            display_error("Prediction Error", f"An error occurred during prediction: {str(e)}")

# Function to display saved results
def display_results(file_path, title):
    try:
        df = pd.read_csv(file_path)
        result_window = tk.Toplevel(root)
        result_window.title(title)
        result_window.geometry("800x400")

        # Create Text widget for displaying results
        result_text = tk.Text(result_window, wrap="none", height=20, width=100)
        result_text.pack(padx=10, pady=10)
        result_text.insert(tk.END, df.head(10).to_string(index=False))  # Display top 10 rows
        result_text.config(state=tk.DISABLED)
    except Exception as e:
        display_error("Error", f"Failed to load the results file: {str(e)}")

# About Section
def about():
    display_message("About", "CustomerInsightPredictor\n\nA project for segmentation and sales prediction.")

# Contact Section
def contact():
    display_message("Contact", "For inquiries, contact:\nEmail: support@customerinsight.com")

# Main Application Window
root = tk.Tk()
root.title("Customer Insight Predictor")
root.geometry("800x700")
root.configure(bg="#f7f7f7")  # Light background color

# Title Bar
title_frame = tk.Frame(root, bg="#8ecae6", height=100)
title_frame.pack(fill="x")

title_label = tk.Label(
    title_frame,
    text="Customer Insight Predictor",
    font=("Verdana", 24, "bold"),
    bg="#8ecae6",
    fg="#023047",
)
title_label.pack(pady=25)

# Segmentation Section
segmentation_frame = tk.Frame(root, bg="#caf0f8", padx=20, pady=20, highlightbackground="#023047", highlightthickness=1)
segmentation_frame.pack(pady=15, padx=30, fill="x")

seg_label = tk.Label(segmentation_frame, text="Segmentation", font=("Verdana", 16, "bold"), bg="#caf0f8", fg="#023047")
seg_label.pack(anchor="w")

seg_description = tk.Label(
    segmentation_frame,
    text="Analyze and segment customer data using KMeans clustering.",
    font=("Verdana", 12),
    bg="#caf0f8",
    fg="#023047",
)
seg_description.pack(anchor="w", pady=5)

seg_button = tk.Button(
    segmentation_frame,
    text="Browse Dataset & Run Segmentation",
    command=segmentation,
    bg="#8ecae6",
    fg="#023047",
    font=("Verdana", 12),
    padx=10,
    pady=5,
    relief="flat",
    activebackground="#023047",
    activeforeground="white",
)
seg_button.pack(pady=10)

# Prediction Section
prediction_frame = tk.Frame(root, bg="#caf0f8", padx=20, pady=20, highlightbackground="#023047", highlightthickness=1)
prediction_frame.pack(pady=15, padx=30, fill="x")

pred_label = tk.Label(prediction_frame, text="Prediction", font=("Verdana", 16, "bold"), bg="#caf0f8", fg="#023047")
pred_label.pack(anchor="w")

pred_description = tk.Label(
    prediction_frame,
    text="Predict customer sales or spending behavior using the model.",
    font=("Verdana", 12),
    bg="#caf0f8",
    fg="#023047",
)
pred_description.pack(anchor="w", pady=5)

pred_button = tk.Button(
    prediction_frame,
    text="Browse Dataset & Run Prediction",
    command=prediction,
    bg="#8ecae6",
    fg="#023047",
    font=("Verdana", 12),
    padx=10,
    pady=5,
    relief="flat",
    activebackground="#023047",
    activeforeground="white",
)
pred_button.pack(pady=10)

# Footer Section
footer_frame = tk.Frame(root, bg="#8ecae6", height=50)
footer_frame.pack(fill="x", side="bottom")

about_button = tk.Button(
    footer_frame,
    text="About",
    command=about,
    bg="#023047",
    fg="#f7f7f7",
    font=("Verdana", 12),
    width=12,
    relief="flat",
    activebackground="#219ebc",
    activeforeground="white",
)
about_button.pack(side="left", padx=10, pady=10)

contact_button = tk.Button(
    footer_frame,
    text="Contact",
    command=contact,
    bg="#023047",
    fg="#f7f7f7",
    font=("Verdana", 12),
    width=12,
    relief="flat",
    activebackground="#219ebc",
    activeforeground="white",
)
contact_button.pack(side="right", padx=10, pady=10)

# Run the application
root.mainloop()
