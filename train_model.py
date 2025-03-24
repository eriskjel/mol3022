import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# Constants
WINDOW_SIZE = 17  # Use 17 amino acids (8 on each side of the central amino acid)
AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
SS_LETTERS = "_he"  # Coil (_), helix (h), sheet (e)


# Load data files
def load_data(filename):
    sequences = []
    structures = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    current_seq = ""
    current_struct = ""

    for line in lines:
        line = line.strip()
        if not line:
            if current_seq:
                sequences.append(current_seq)
                structures.append(current_struct)
                current_seq = ""
                current_struct = ""
        else:
            # This format may need adjustment based on the actual file format
            parts = line.split()
            if len(parts) >= 2:  # Assuming format: amino_acid structure
                current_seq += parts[0]
                current_struct += parts[1]

    if current_seq:  # Add the last sequence
        sequences.append(current_seq)
        structures.append(current_struct)

    return sequences, structures


# One-hot encode amino acids
def one_hot_encode_aa(aa):
    """Convert amino acid letter to one-hot encoding."""
    encoding = np.zeros(len(AA_LETTERS))
    if aa in AA_LETTERS:
        encoding[AA_LETTERS.index(aa)] = 1
    return encoding


# Create windows from sequences
def create_windows(sequences, structures, window_size):
    """
    Create sliding windows from sequences and their corresponding structure labels.

    Parameters:
    sequences -- List of amino acid sequences
    structures -- List of secondary structure sequences
    window_size -- Size of the window (odd number)

    Returns:
    X -- List of encoded windows
    y -- List of corresponding structure labels (for the central amino acid)
    """
    half_window = window_size // 2
    X = []
    y = []

    for sequence, structure in zip(sequences, structures):
        for i in range(len(sequence)):
            # Extract window
            window = ''
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
                    encoded_window.extend(one_hot_encode_aa(aa))

            # Add to dataset
            X.append(encoded_window)

            # Encode structure (one-hot)
            if i < len(structure):
                ss = structure[i]
                ss_encoding = np.zeros(len(SS_LETTERS))
                if ss in SS_LETTERS:
                    ss_encoding[SS_LETTERS.index(ss)] = 1
                y.append(ss_encoding)
            else:
                # This should not happen if sequence and structure have the same length
                print(f"Warning: structure not found for position {i} in sequence")

    return np.array(X), np.array(y)


def train_and_save_model(train_file, test_file, output_model, window_size=WINDOW_SIZE):
    # Load data
    print("Loading data...")
    train_seq, train_struct = load_data(train_file)
    test_seq, test_struct = load_data(test_file)

    # Basic statistics
    print(f"Training set: {len(train_seq)} proteins, {sum(len(s) for s in train_seq)} amino acids")
    print(f"Test set: {len(test_seq)} proteins, {sum(len(s) for s in test_seq)} amino acids")

    # Create window datasets
    print(f"Creating window datasets with window size {window_size}...")
    X_train, y_train = create_windows(train_seq, train_struct, window_size)
    X_test, y_test = create_windows(test_seq, test_struct, window_size)

    print(f"Training examples: {X_train.shape[0]}")
    print(f"Test examples: {X_test.shape[0]}")

    # Create a validation set from the training data
    X_train_subset, X_val, y_train_subset, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print(f"Training subset examples: {X_train_subset.shape[0]}")
    print(f"Validation examples: {X_val.shape[0]}")

    # Build model
    print("Building neural network model...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(SS_LETTERS), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Train model
    print("Training model...")
    history = model.fit(
        X_train_subset, y_train_subset,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")

    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Calculate metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Coil', 'Helix', 'Sheet']))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Coil', 'Helix', 'Sheet'],
                yticklabels=['Coil', 'Helix', 'Sheet'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('training_history.png')

    # Save model
    print(f"Saving model to {output_model}")
    model.save(output_model)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train a protein secondary structure prediction model')
    parser.add_argument('--train', type=str,
                        default='data/protein-secondary-structure.train',
                        help='Training data file')
    parser.add_argument('--test', type=str,
                        default='data/protein-secondary-structure.test',
                        help='Test data file')
    parser.add_argument('--output', type=str,
                        default='protein_ss_model.h5',
                        help='Output model file')
    parser.add_argument('--window', type=int,
                        default=WINDOW_SIZE,
                        help='Window size')
    args = parser.parse_args()

    train_and_save_model(args.train, args.test, args.output, args.window)


if __name__ == "__main__":
    main()