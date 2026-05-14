import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SmallCNN

# ── Hyperparamètres ──────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data augmentation + normalisation ────────────────────────────────────────
# Moyenne et std calculées sur CIFAR-10 (valeurs standard, à connaître)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),          # augmentation simple
    transforms.RandomCrop(32, padding=4),       # augmentation standard CIFAR
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    # Pas d'augmentation au test — on évalue sur données propres
])

# ── Chargement des données ────────────────────────────────────────────────────
train_set = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
test_set  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── Modèle, loss, optimiseur ─────────────────────────────────────────────────
model     = SmallCNN(num_classes=10).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Scheduler : réduit le LR x10 aux epochs 10 et 16
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 16], gamma=0.1)

# ── Fonctions train / eval ────────────────────────────────────────────────────
def train_one_epoch(epoch: int) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()           # remet les gradients à zéro
        outputs = model(images)         # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                 # backward pass (calcul des gradients)
        optimizer.step()               # mise à jour des poids

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx*BATCH_SIZE}/{len(train_set)}] "
                  f"Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)  # loss moyenne sur l'epoch


def evaluate() -> float:
    model.eval()
    correct = 0

    with torch.no_grad():   # désactive le calcul des gradients pendant l'éval
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)     # classe prédite = argmax des logits
            correct += (preds == labels).sum().item()

    return correct / len(test_set)


# ── Boucle principale ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Training on {DEVICE}")
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(epoch)
        test_acc   = evaluate()
        scheduler.step()

        print(f"── Epoch {epoch:02d} | Loss: {train_loss:.4f} | "
              f"Test Acc: {test_acc*100:.2f}%")

        # Sauvegarde du meilleur modèle
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"   ✓ New best model saved ({best_acc*100:.2f}%)")

    print(f"\nFinal best accuracy: {best_acc*100:.2f}%")