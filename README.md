# CIFAR-10 Classifier — PyTorch from scratch

Small CNN trained on CIFAR-10 without pretrained weights.

## Results
| Model     | Test Accuracy | Epochs |
|-----------|--------------|--------|
| SmallCNN  | ~74%         | 20     |

## Architecture
3 conv blocks (Conv, BN, ReLU, MaxPool) + MLP classifier.

## Run
pip install -r requirements.txt
python train.py

## Key learnings
- Manual PyTorch training loop (no high-level API)
- BatchNorm + Dropout regularization
- LR scheduling with MultiStepLR