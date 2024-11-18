import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
import pandas as pd
import time
import logging

# Add parent directory to module search path
sys.path.append("C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor")

from src.segmentation_model import perform_segmentation
from src.prediction_model import perform_prediction

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to browse file
def browse_file(filetypes, initial_dir, title):
    return filedialog.askopenfilename(
        title=title,
        filetypes=filetypes,
        initialdir=initial_dir
    )

# Function to display messages
def display_message(title, message):
    messagebox.showinfo(title, message)

def display_error(title, message):
    messagebox.showerror(title, message)

# Functionality for Segmentation
def segmentation():
    file_path = browse_file([("CSV Files", "*.csv")], "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/data/", "Select Mall_Customers.csv")
    
    if file_path:
        output_path = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/segmentation_results.csv"
        
        start_time = time.time()  # Start timer
        perform_segmentation(file_path, output_path)
        end_time = time.time()  # End timer
        
        execution_time = end_time - start_time
        display_message("Segmentation Complete", f"Results saved to {output_path}\nTime taken: {execution_time:.2f} seconds")

        # Show the saved results
        show_results(output_path)

# Functionality for Prediction
def prediction():
    file_path = browse_file([("CSV Files", "*.csv")], "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/data/", "Select sales_prediction_dataset.csv")
    
    if file_path:
        output_path = "C:/Users/DELL/OneDrive/Desktop/CustomerInsightPredictor/results/prediction_result.csv"
        
        try:
            # Create a configuration dictionary
            config = {
                "file_path": file_path,
                "output_path": output_path
            }
            
            start_time = time.time()  # Start timer
            logging.info(f"Starting prediction with config: {config}")  # Debugging
            perform_prediction(config)  # Pass the config to the updated perform_prediction function
            end_time = time.time()  # End timer
            
            execution_time = end_time - start_time
            display_message("Prediction Complete", f"Results saved to {output_path}\nTime taken: {execution_time:.2f} seconds")
            
            # Show the saved results
            show_results(output_path)
        except Exception as e:
            display_error("Prediction Error", f"An error occurred during prediction: {str(e)}")

# Function to display saved results
def show_results(file_path):
    try:
        df = pd.read_csv(file_path)
        result_window = tk.Toplevel(root)
        result_window.title("Results")
        result_window.geometry("700x400")
        
        # Create a Text widget to display the dataframe
        result_text = tk.Text(result_window, wrap="none", height=20, width=80)
        result_text.pack(padx=10, pady=10)

        # Insert the dataframe into the Text widget
        result_text.insert(tk.END, df.to_string())
        result_text.config(state=tk.DISABLED)  # Make the text widget read-only
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
root.geometry("700x600")
root.configure(bg="#1e1e2f")  # Dark background color

# Title Bar
title_frame = tk.Frame(root, bg="#4b6584", height=100)
title_frame.pack(fill="x")

title_label = tk.Label(
    title_frame,
    text="Customer Insight Predictor",
    font=("Verdana", 24, "bold"),
    bg="#4b6584",
    fg="#f5f6fa",
)
title_label.pack(pady=25)

# Segmentation Section
segmentation_frame = tk.Frame(root, bg="#2f3640", padx=20, pady=20, highlightbackground="#dcdde1", highlightthickness=1)
segmentation_frame.pack(pady=15, padx=30, fill="x")

seg_label = tk.Label(segmentation_frame, text="Segmentation", font=("Verdana", 16, "bold"), bg="#2f3640", fg="#00a8ff")
seg_label.pack(anchor="w")

seg_description = tk.Label(
    segmentation_frame,
    text="Analyze and segment customer data using KMeans clustering.",
    font=("Verdana", 12),
    bg="#2f3640",
    fg="#dcdde1",
)
seg_description.pack(anchor="w", pady=5)

seg_button = tk.Button(
    segmentation_frame,
    text="Browse Dataset & Run Segmentation",
    command=segmentation,
    bg="#44bd32",
    fg="#f5f6fa",
    font=("Verdana", 12),
    padx=10,
    pady=5,
    relief="flat",
    activebackground="#4cd137",
    activeforeground="white",
)
seg_button.pack(pady=10)

# Prediction Section
prediction_frame = tk.Frame(root, bg="#2f3640", padx=20, pady=20, highlightbackground="#dcdde1", highlightthickness=1)
prediction_frame.pack(pady=15, padx=30, fill="x")

pred_label = tk.Label(prediction_frame, text="Prediction", font=("Verdana", 16, "bold"), bg="#2f3640", fg="#00a8ff")
pred_label.pack(anchor="w")

pred_description = tk.Label(
    prediction_frame,
    text="Predict customer sales or spending behavior using the model.",
    font=("Verdana", 12),
    bg="#2f3640",
    fg="#dcdde1",
)
pred_description.pack(anchor="w", pady=5)

pred_button = tk.Button(
    prediction_frame,
    text="Browse Dataset & Run Prediction",
    command=prediction,
    bg="#44bd32",
    fg="#f5f6fa",
    font=("Verdana", 12),
    padx=10,
    pady=5,
    relief="flat",
    activebackground="#4cd137",
    activeforeground="white",
)
pred_button.pack(pady=10)

# Footer Section
footer_frame = tk.Frame(root, bg="#4b6584", height=100)
footer_frame.pack(fill="x", side="bottom")

about_button = tk.Button(
    footer_frame,
    text="About",
    command=about,
    bg="#40739e",
    fg="#f5f6fa",
    font=("Verdana", 12),
    width=12,
    relief="flat",
    activebackground="#487eb0",
    activeforeground="white",
)
about_button.pack(side="left", padx=10, pady=20)

contact_button = tk.Button(
    footer_frame,
    text="Contact",
    command=contact,
    bg="#40739e",
    fg="#f5f6fa",
    font=("Verdana", 12),
    width=12,
    relief="flat",
    activebackground="#487eb0",
    activeforeground="white",
)
contact_button.pack(side="right", padx=10, pady=20)

# Run the application
root.mainloop()
