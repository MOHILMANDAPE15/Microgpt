# Microgpt
# MicroGPT

MicroGPT is a simplified transformer-based language model designed to generate text and learn from input data. This project demonstrates the implementation of a miniature GPT-like model using PyTorch, complete with training, validation, and text generation functionality.

---

## Features
- Implements key GPT components, including self-attention and feedforward layers.
- Supports tokenization with [tiktoken](https://github.com/openai/tiktoken) (optimized for GPT models).
- Trains on custom text datasets and evaluates performance.
- Includes text generation with temperature and top-k sampling for enhanced control over output diversity.

---

## Requirements

- Python 3.8+
- PyTorch
- tiktoken

Install dependencies with:
```bash
pip install torch tiktoken
```

---

## How It Works

### Model Components

1. **Feed-Forward Network (FFN):**
   A two-layer feed-forward network with ReLU activation and dropout for regularization.

2. **Self-Attention Mechanism:**
   Masked self-attention ensures the model only considers prior tokens during training.

3. **Multiple Attention Heads:**
   Combines multiple self-attention layers to capture richer patterns in data.

4. **Transformer Blocks:**
   Stacked transformer layers to encode contextual information.

5. **MicroGPT Architecture:**
   Combines embedding layers, positional encoding, transformer blocks, and a final linear layer to output token logits.

---

## Code Structure

1. **Data Preparation:**
   - Tokenizes the input text using `tiktoken`.
   - Splits data into training and validation sets.

2. **Model Training:**
   - Uses `cross-entropy` loss for optimization.
   - Trains for a defined number of epochs and saves the best-performing model.

3. **Text Generation:**
   - Generates text using the trained model.
   - Supports temperature scaling and top-k sampling for diverse outputs.

---

## Usage

### Training the Model

1. Place your training data in a text file, e.g., `input.txt`.
2. Adjust hyperparameters as needed (e.g., `n_embed`, `n_ma`, `lr`, `epochs`).
3. Run the script:

```bash
python microgpt.py
```

### Generating Text

After training, the script generates text using the following function:
```python
generate_text(model, start_text="hello", length=2000)
```
Replace `start_text` and `length` as desired.

---

## Hyperparameters

- **n_embed:** Embedding size (default: 512).
- **n_ma:** Number of attention heads (default: 8).
- **text_size:** Context window size (default: 512).
- **batches:** Batch size (default: 8).
- **lr:** Learning rate (default: 3e-3).
- **epochs:** Number of training epochs (default: 390).

---

## Example Output

Start Text:
```
hello
```

Sample of Generated Text:
```
BRUTLAND: yet? To sweet pleasing markting noble health speak oy's thunder, fatherHun, we unrel case wherein come
Why I say lofty surlevanch contested of pur blot.' what repy married! that given stumbledess, minister, spite? worser thereto, especially.

```

---

## Checkpoints

The best model checkpoint is saved at:
```
/content/best_microgpt.pth
```

Load the model with:
```python
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
```

---

## License

This project is licensed under the MIT License. Feel free to use and modify it for your own projects.

---

## Acknowledgments

- [OpenAI](https://openai.com) for the GPT architecture inspiration.
- [tiktoken](https://github.com/openai/tiktoken) for tokenization.

