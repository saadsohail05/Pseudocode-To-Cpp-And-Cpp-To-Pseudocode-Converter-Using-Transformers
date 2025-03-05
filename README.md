# Pseudocode-To-C++ and C++-To-Pseudocode Converter Using Transformers

A bidirectional converter that translates between C++ code and pseudocode using transformer-based neural networks. The project implements both code-to-pseudocode and pseudocode-to-code translation capabilities using PyTorch.

## Features

- Bidirectional translation between C++ code and pseudocode
- Interactive web interface using Streamlit
- Trained on a comprehensive dataset of code-pseudocode pairs
- Support for common programming constructs:
  - Control structures (if-else, loops)
  - Function declarations
  - Input/output operations
  - Basic arithmetic and logical operations

## Project Structure

```
├── app.py                  # Streamlit web application
├── requirements.txt        # Project dependencies
├── spoc_train.csv         # Training dataset
├── code_to_pseudocode/    # Code to Pseudocode translation model
│   ├── model.py
│   ├── code_tokenizer.model
│   ├── text_tokenizer.model
│   └── code_psd_transformer.pt
├── pseudocode_to_code/    # Pseudocode to Code translation model
│   ├── model.py
│   ├── code_tokenizer.model
│   ├── text_tokenizer.model
│   └── psd_code_transformer.pt
└── Notebooks/             # Development and training notebooks
    ├── Code to PseudoCode.ipynb
    └── PseudoCode to Code.ipynb
```

## Technical Details

- **Architecture**: Transformer-based sequence-to-sequence model
- **Components**:
  - Multi-head attention mechanism
  - Position-wise feed-forward networks
  - Residual connections and layer normalization
  - SentencePiece tokenization

## Installation

1. Clone the repository:
```bash
git clone <https://github.com/saadsohail05/Pseudocode-To-Cpp-And-Cpp-To-Pseudocode-Converter-Using-Transformers-url>
cd Pseudocode-To-Cpp-And-Cpp-To-Pseudocode-Converter-Using-Transformers
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web application:
```bash
streamlit run app.py
```

2. Use the interactive interface to:
   - Convert C++ code to pseudocode
   - Convert pseudocode to C++ code
   - View example translations

## Model Training

The models were trained on a dataset of C++ code and corresponding pseudocode pairs. The training process involved:
- Data preprocessing and tokenization using SentencePiece
- Training separate transformer models for each direction
- Optimization using AdamW optimizer with cosine annealing
- Validation using BLEU score metrics

## Dependencies

- PyTorch
- Streamlit
- SentencePiece
- NumPy
- pandas
- tqdm
- sacrebleu

## Examples

### C++ to Pseudocode:
```cpp
for(int i=0; i<n; i++) { 
    sum += arr[i]; 
}
```
↓ Converts to ↓
```
loop from 0 to n-1:
    add arr[i] to sum
```

### Pseudocode to C++:
```
if x is greater than y then
    set max to x
else
    set max to y
```
↓ Converts to ↓
```cpp
if(x > y) {
    max = x;
} else {
    max = y;
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

