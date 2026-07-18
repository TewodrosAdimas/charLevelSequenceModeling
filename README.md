# Character-Level Text Generation using CNN, RNN, LSTM, GRU, and BiLSTM

A deep learning project that implements and compares multiple neural network architectures for **character-level text generation**. The models learn the statistical patterns of text from *Alice's Adventures in Wonderland* and generate new text one character at a time.

The project investigates how different sequence models—including CNNs, vanilla RNNs, GRUs, LSTMs, and Bidirectional LSTMs—perform on the task of next-character prediction.

---

## 📖 Project Overview

Character-level language models predict the next character given a sequence of previous characters. Unlike word-level language models, character-based models learn spelling, punctuation, and sentence structure directly from raw text without requiring tokenization.

In this project, several neural network architectures are trained and evaluated using the same dataset and preprocessing pipeline to compare their learning capabilities.

---

## 🎯 Objectives

- Build a character-level dataset using sliding windows.
- Train multiple deep learning architectures.
- Compare CNN and recurrent neural networks.
- Evaluate prediction accuracy and loss.
- Visualize confusion matrices.
- Analyze the effect of different LSTM configurations.
- Generate text in the writing style of *Alice's Adventures in Wonderland*.

---

## 🧠 Models Implemented

- **CharCNN**
  - 1D Convolutional Neural Network
  - Max Pooling
  - Fully Connected Layers

- **Vanilla RNN**
  - Two-layer recurrent neural network
  - Tanh activation

- **GRU**
  - Two-layer Gated Recurrent Unit

- **LSTM**
  - Standard two-layer LSTM

- **Deep LSTM**
  - Four LSTM layers
  - Larger hidden dimension

- **Small LSTM**
  - Lightweight single-layer LSTM

- **Bidirectional LSTM**
  - Processes text in both directions
  - Richer contextual representation

---

## 📚 Dataset

**Book**

> Alice's Adventures in Wonderland  
> Author: Lewis Carroll

Source:

Project Gutenberg

---

## Data Preprocessing

The preprocessing pipeline includes:

- Convert text to lowercase
- Remove punctuation and special characters
- Keep only alphabetic characters
- Build character vocabulary
- Character-to-index encoding
- Sliding window sequence generation

Sequence length:

```
200 characters
```

Target:

```
Predict character 201
```

Vocabulary size:

```
27 unique characters
```

---

## Project Structure

```
Character-Level-Text-Generation/
│
├── wonderland.txt
├── cnn_best.pth
├── lstm_best.pth
├── gru_best.pth
├── rnn_best.pth
├── bilstm_best.pth
├── lstm_deep_best.pth
├── lstm_small_best.pth
│
├── notebook.ipynb
│
├── README.md
│
└── figures/
    ├── loss_curves.png
    ├── accuracy_curves.png
    ├── cnn_confusion.png
    ├── lstm_confusion.png
    └── ...
```

---

## ⚙️ Technologies Used

- Python
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Google Colab

---

## Installation

Clone the repository

```bash
git clone https://github.com/TewodrosAdimas/charLevelSequenceModeling

cd character-text-generation
```

Install dependencies

```bash
pip install torch torchvision
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install tqdm
```

---

## Running the Project

Run the notebook

```bash
jupyter notebook
```

or

```bash
python train.py
```

(if converted into Python scripts)

---

## Training Configuration

| Parameter | Value |
|-----------|--------|
| Sequence Length | 200 |
| Batch Size | 128 |
| Epochs | 60 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Early Stopping | Yes |
| Checkpoint Saving | Yes |

---

## Evaluation Metrics

The models are evaluated using

- Cross Entropy Loss
- Character Prediction Accuracy
- Confusion Matrix
- Training Loss Curves
- Validation Loss Curves

---

## Model Comparison

The following architectures were compared:

| Model | Description |
|--------|-------------|
| CNN | Learns local character patterns using convolutions |
| RNN | Basic recurrent model |
| GRU | Faster recurrent architecture with gating |
| LSTM | Standard long-term dependency model |
| Deep LSTM | Larger network with additional layers |
| Small LSTM | Lightweight architecture |
| BiLSTM | Bidirectional contextual learning |

---

## Training Features

The training pipeline includes

- GPU support (when available)
- Mini-batch training
- Model checkpointing
- Early stopping
- Automatic validation
- Confusion matrix generation
- Loss tracking

---

## Example Workflow

```
Raw Text
      │
      ▼
Cleaning
      │
      ▼
Character Encoding
      │
      ▼
Sliding Window Dataset
      │
      ▼
Train Neural Networks
      │
      ▼
Validation
      │
      ▼
Model Comparison
      │
      ▼
Character Prediction
      │
      ▼
Text Generation
```

---

## Sample Results

The project compares models based on validation loss and prediction accuracy.

Example observations:

- CNN learns local character patterns efficiently but struggles with long-range dependencies.
- RNN captures sequential information but suffers from vanishing gradients.
- GRU provides a good balance between speed and performance.
- LSTM improves long-term memory retention.
- Deep LSTM increases model capacity.
- Bidirectional LSTM achieved the best overall performance by utilizing context from both directions.

---

## Learning Outcomes

This project demonstrates

- Character-level language modeling
- Sequence prediction
- Neural language models
- CNN versus recurrent architectures
- Long-term dependency learning
- Hyperparameter experimentation
- Model evaluation techniques
- Text generation using deep learning

---

## Future Improvements

- Transformer-based language models
- Attention mechanisms
- Temperature-controlled sampling
- Beam search decoding
- Word-level language modeling
- Byte Pair Encoding (BPE)
- Larger datasets
- Mixed precision GPU training
- TensorBoard integration

---

## References

- PyTorch Documentation
- Project Gutenberg
- Hochreiter & Schmidhuber (1997) – Long Short-Term Memory
- Cho et al. (2014) – GRU Networks
- Goodfellow et al. – Deep Learning

---

## Author

**Tewodros Bewuket Adimas**

MSc in Artificial Intelligence for Science and Technology

University of Milano-Bicocca  
University of Milan  
University of Pavia

---

## License

This project is intended for educational and research purposes.
