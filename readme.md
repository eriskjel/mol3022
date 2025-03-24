# Protein Secondary Structure Prediction

A neural network-based tool for predicting protein secondary structure from amino acid sequences.

## Background

Protein secondary structure refers to the local conformations of the polypeptide backbone. This tool predicts three types of secondary structure:
- Coil/loop (_)
- Alpha-helix (h)
- Beta-sheet (e)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/protein-ss-prediction.git
cd protein-ss-prediction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt