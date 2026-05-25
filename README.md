# LSTM Name Generator

A character-level LSTM trained on the SSA Baby Names dataset that generates new, realistic-sounding names by learning patterns from real ones.

## How it works

The model processes names one character at a time, learning transitions like:
- Names often end in `-a`, `-on`, `-ey`, `-elle`
- Common letter combos like `ar`, `an`, `el`, `ly`
- Realistic name lengths

At generation time, it feeds one character at a time in a feedback loop — the output of each step becomes the input of the next — until an `<end>` token is produced.

## Model Architecture

```
Character Index → Embedding (28x32) → LSTM (128 hidden) → Linear (128x28) → 28 character scores
```

- **Embedding layer** — converts character indices to 32-dimensional vectors
- **LSTM** — processes sequences, maintaining hidden state across characters
- **Linear layer** — maps LSTM output to scores over the 28-character vocabulary (26 letters + `<start>` + `<end>`)

## Dataset

Uses the [SSA Baby Names dataset](https://www.ssa.gov/oact/babynames/limits.html) — a public domain dataset of US baby names from 1880 to present. Used the 2010 dataset for this particular model, as it is more relevant

## Setup

```bash
conda create -n nameenv python=3.11
conda activate nameenv
pip install torch pandas
```

## Usage

```bash
python model.py
```

You will be prompted to enter:
- Number of names to generate
- Maximum length of each name

## Sample Output

```
shannon
kylie
beatrice
aryana
amaya
rebekah
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dim | 32 |
| Hidden dim | 128 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Epochs | 200 |
| Temperature | 0.8 |

## File Structure

```
├── data/
│   └── yob2010.txt        # SSA names dataset
├── name.py                # model + training + generation
└── README.md
```