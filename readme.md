# Protein Secondary Structure Prediction Tool


This tool uses a deep learning approach to predict the secondary structure of proteins based solely on their amino acid sequences. The prediction model is implemented as a feedforward neural network that analyzes local amino acid patterns to classify each residue as part of an alpha-helix, beta-sheet, or random coil.
## Background

Protein secondary structure refers to the local conformations of the polypeptide backbone. This tool predicts three types of secondary structure:
- Coil/loop (_)
- Alpha-helix (h)
- Beta-sheet (e)

## Installation

```
# Clone the repository
git clone https://github.com/eriskjel/mol3022
cd mol3022

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## Usage

The easiest way to use this tool is through the graphical interface:
```
python predict_ss.py
```

This will open the application window where you can:

1. Enter a protein sequence directly in the text box
2. Load a sequence from a file using the "Load Sequence" button
3. Generate a random sequence for testing using the "Generate Random" button
4. Click "Predict Structure" to analyze the sequence
5. View the results in the output area and visualization panel

## Training a New Model (Optional)
The repository includes a pre-trained model, but you can train your own if desired:
````
python train_model.py --train data/protein-secondary-structure.train --test data/protein-secondary-structure.test --output protein_ss_model.h5
````

### Arguments:

- --train: Training data file (default: data/protein-secondary-structure.train)
- --test: Test data file (default: data/protein-secondary-structure.test)
- --output: Output model file name (default: protein_ss_model.h5)
- --window: Window size for sequence analysis (default: 17)