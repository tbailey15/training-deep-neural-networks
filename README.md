# Training Deep Neural Networks

An iris flower classifier in PyTorch that uses **dropout**, **weight decay**, and
**early stopping**, plus a test of whether dropout helps.

Everything is in [`training_deep_neural_networks.ipynb`](training_deep_neural_networks.ipynb).

## Results

| | Result |
|---|---|
| Baseline (always guess one species) | 33.3% |
| Validation, dropout 0.2 | loss 0.154, accuracy 93.3% |
| Validation, dropout 0.0 | loss 0.215, accuracy 93.3% |
| Chosen setting (lower validation loss) | dropout 0.2 |
| **Final test accuracy** | **96.7% (29 of 30)** |

- Data split: 90 training, 30 validation, 30 test flowers.
- Network: 4 → 64 → 64 → 3, with ReLU and dropout. Adam (lr 0.01, weight decay 0.001).
- Early stopping (patience 15) kept the best weights: epoch 64 with dropout and
  epoch 11 without it.
- The test set was used only once, after the choice was made. With 30 test flowers,
  one mistake is worth about 3.3 points, so this is a small check.

## How to run it

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

```bash
jupyter nbconvert --to notebook --execute --inplace training_deep_neural_networks.ipynb
```

Or open the notebook and click **Run All**. Keep `iris_demo.csv` in the same folder.

## Data

Fisher's iris dataset (150 flowers, 4 measurements in cm, 3 species), from
scikit-learn's copy.

## Reference

Géron, A. (2025). *Hands-On Machine Learning with Scikit-Learn and PyTorch.*
O'Reilly Media. (Chapter 11)
