#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model

# Constants
WINDOW_SIZE = 17
AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
SS_LETTERS = "_he"  # Coil, helix, sheet


class ProteinStructurePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Protein Secondary Structure Predictor")
        self.root.geometry("800x600")

        try:
            self.model = load_model('protein_ss_model.h5')
            self.model_loaded = True
        except:
            try:
                # Try relative path as fallback
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, 'protein_ss_model.h5')
                self.model = load_model(model_path)
                self.model_loaded = True
            except:
                self.model_loaded = False

        self.create_widgets()

    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create input frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Sequence", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sequence input
        self.sequence_text = scrolledtext.ScrolledText(input_frame, height=5)
        self.sequence_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Button frame
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        # Load sequence button
        load_button = ttk.Button(button_frame, text="Load Sequence", command=self.load_sequence)
        load_button.pack(side=tk.LEFT, padx=5)

        # Generate random sequence button
        random_button = ttk.Button(button_frame, text="Generate Random", command=self.generate_random)
        random_button.pack(side=tk.LEFT, padx=5)

        # Predict button
        predict_button = ttk.Button(button_frame, text="Predict Structure", command=self.predict)
        predict_button.pack(side=tk.LEFT, padx=5)

        # Clear button
        clear_button = ttk.Button(button_frame, text="Clear", command=self.clear)
        clear_button.pack(side=tk.LEFT, padx=5)

        # Model status
        if self.model_loaded:
            status_text = "Model loaded successfully"
            status_color = "green"
        else:
            status_text = "Error: Model not found. Please train the model first."
            status_color = "red"

        status_label = ttk.Label(button_frame, text=status_text, foreground=status_color)
        status_label.pack(side=tk.RIGHT, padx=5)

        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Results text
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure for visualization
        self.fig, self.ax = plt.subplots(figsize=(8, 2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_sequence(self):
        file_path = filedialog.askopenfilename(
            title="Select Sequence File",
            filetypes=(("FASTA files", "*.fasta"), ("Text files", "*.txt"), ("All files", "*.*"))
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                sequence = ""
                for line in lines:
                    if not line.startswith('>'):  # Skip FASTA header
                        sequence += line.strip()

                self.sequence_text.delete(1.0, tk.END)
                self.sequence_text.insert(tk.END, sequence)
            except Exception as e:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, f"Error loading file: {str(e)}")

    def generate_random(self):
        # Generate a random protein sequence
        length = random.randint(30, 100)
        sequence = ''.join(random.choice(AA_LETTERS) for _ in range(length))

        self.sequence_text.delete(1.0, tk.END)
        self.sequence_text.insert(tk.END, sequence)

    def predict(self):
        if not self.model_loaded:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Error: Model not loaded. Please train the model first.")
            return

        sequence = self.sequence_text.get(1.0, tk.END).strip()
        if not sequence:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Error: Please enter a protein sequence.")
            return

        # Predict structure
        predicted_structure = self.predict_structure(sequence)

        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Sequence: {sequence}\n")
        self.results_text.insert(tk.END, f"Predicted: {predicted_structure}\n\n")

        # Format alignment
        self.results_text.insert(tk.END, "Alignment:\n")
        for i in range(0, len(sequence), 80):
            end = min(i + 80, len(sequence))
            self.results_text.insert(tk.END, f"{sequence[i:end]}\n")
            self.results_text.insert(tk.END, f"{predicted_structure[i:end]}\n\n")

        # Visualize
        self.visualize_prediction(sequence, predicted_structure)

    def clear(self):
        self.sequence_text.delete(1.0, tk.END)
        self.results_text.delete(1.0, tk.END)
        self.ax.clear()
        self.canvas.draw()

    def one_hot_encode_aa(self, aa):
        """Convert amino acid letter to one-hot encoding."""
        encoding = np.zeros(len(AA_LETTERS))
        if aa in AA_LETTERS:
            encoding[AA_LETTERS.index(aa)] = 1
        return encoding

    def predict_structure(self, sequence):
        """Predict secondary structure for a protein sequence"""
        half_window = WINDOW_SIZE // 2
        predicted_structure = ""

        for i in range(len(sequence)):
            # Extract window
            window = ""
            for j in range(i - half_window, i + half_window + 1):
                if j < 0 or j >= len(sequence):
                    window += '-'  # Padding for positions outside sequence
                else:
                    window += sequence[j]

            # Encode window
            encoded_window = []
            for aa in window:
                if aa == '-':
                    encoded_window.extend(np.zeros(len(AA_LETTERS)))
                else:
                    encoded_window.extend(self.one_hot_encode_aa(aa))

            # Make prediction
            encoded_window = np.array([encoded_window])
            prediction = self.model.predict(encoded_window, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            predicted_structure += SS_LETTERS[predicted_class]

        return predicted_structure

    def visualize_prediction(self, sequence, predicted_structure):
        """Create a visualization of the predicted structure"""
        self.ax.clear()

        # Define colors for each structure type
        colors = {'_': 'gray', 'h': 'red', 'e': 'blue'}

        # Plot the structure
        for i, ss in enumerate(predicted_structure):
            self.ax.plot([i, i + 1], [0, 0], color=colors.get(ss, 'black'), linewidth=10)

        # Add amino acid labels (if sequence is not too long)
        if len(sequence) <= 100:
            for i, aa in enumerate(sequence):
                self.ax.text(i + 0.5, -0.3, aa, horizontalalignment='center', fontsize=8)

        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Alpha-helix (h)'),
            Patch(facecolor='blue', label='Beta-sheet (e)'),
            Patch(facecolor='gray', label='Coil (_)')
        ]
        self.ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

        # Set layout
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_xlim(0, len(sequence))
        self.ax.axis('off')
        self.ax.set_title('Secondary Structure Prediction')

        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ProteinStructurePredictorApp(root)
    root.mainloop()
