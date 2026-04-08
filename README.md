# Name Generator — Neural Language Models from Scratch

A progressive implementation of character-level neural language models that learn to generate human-like names. Built from scratch using PyTorch, following Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) series for clear and intuitive understanding of AI systems.

## Dataset

The model trains on **~32,000 human names** (`names.txt`). Each name is broken into character sequences that the model learns to predict, ultimately generating novel, realistic-sounding names.

## Notebooks

Each notebook builds on the previous one, introducing new concepts incrementally:

### 1. Bigrams (`Bigrams.ipynb`)
The simplest language model — counts how often one character follows another and samples from those frequencies. No neural network involved, just statistics.

### 2. Character-Level MLP (`charGenerator.ipynb`)
A multi-layer perceptron following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf):
- **Character embeddings** (27 chars → 10-dimensional vectors)
- **Context window** of 3 characters to predict the next one
- Single hidden layer (200 neurons) with tanh activation
- Trained with minibatch SGD (batch size 128, 200K steps)
- Achieves ~2.17 train loss / ~2.19 validation loss
- Includes embedding visualization and learning rate finder

### 3. Batch Normalization & Deeper Networks (`wordGenBatchNorm.ipynb`)
Scales up with proper initialization and normalization:
- **Batch Normalization** to stabilize training of deeper networks
- **Kaiming initialization** for proper gradient flow
- **5 hidden layers** (100 neurons each) with custom `Linear`, `BatchNorm1d`, and `Tanh` classes
- **Diagnostic plots**: activation distributions, gradient distributions, weight update-to-data ratios
- Achieves ~2.07 train loss / ~2.10 validation loss

### 4. Manual Backpropagation (`tensorBackPropogation.ipynb`)
Derives and implements backpropagation manually through the entire computation graph — cross-entropy, batch norm, tanh, matrix multiplications, and embeddings — without using `loss.backward()`.

### 5. WaveNet Implementation (`wavenetImpl.ipynb`)
WaveNet-style hierarchical architecture that processes characters in a tree-like fashion:
- **Larger context window** (block_size=8, up from 3)
- **`FlattenConsecutive`** layer that groups pairs of embeddings, enabling hierarchical merging
- **`Sequential`** container and **`Embedding`** class for a cleaner, more modular design
- 3-level hierarchy: pairs of characters are merged progressively (8→4→2→1)
- 76K parameters, 24-dim embeddings, 128 hidden neurons
- Achieves **~1.77 train loss / ~1.99 validation loss** — best results in the series
- Demonstrates that convolutions are essentially efficient "for loops" over spatial positions

## How It Works

```
"names.txt"  →  character sequences  →  sliding window contexts
                                              ↓
                         [ ... ] → [a] → [a,r] → [a,r,i] → predict 'e'
                                              ↓
                              Embedding → Hidden Layers → Softmax → Loss
                                              ↓
                                    SGD updates weights
                                              ↓
                                  Sample new names from model
```

The vocabulary is 27 characters (a–z plus a special `.` token for start/end of name). During training, the model sees a sliding window of characters and learns to predict the next one. At generation time, it samples one character at a time, feeding predictions back as input.

## Requirements

- Python 3.x
- PyTorch
- Matplotlib

```bash
pip install torch matplotlib
```

## Usage

Open any notebook in Jupyter or VS Code and run all cells. No additional setup required — just the notebooks and `names.txt`.

```bash
jupyter notebook
```

## Sample Output

The WaveNet model (notebook 5) generates realistic names like:
```
chetta
hendrix
jamylie
marianah
jayce
jaylene
aubreana
```

Earlier models produce less realistic output — quality improves progressively across notebooks.

## Acknowledgements

Based on Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) lecture series and [Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (Bengio et al., 2003).
