# classifier/train_aircraft_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DATA_ROOT = "data/fgvc_aircraft"
BATCH_SIZE = 64
NUM_EPOCHS = 30
LR = 1e-3
LEVEL = "variant"  # или "family"

def get_dataloaders():
    t_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    t_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.FGVCAircraft(
        root=DATA_ROOT, split="train", download=True,
        transform=t_train, annotation_level=LEVEL
    )
    val_ds = datasets.FGVCAircraft(
        root=DATA_ROOT, split="val", download=True,
        transform=t_val, annotation_level=LEVEL
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)
    return train_loader, val_loader, len(train_ds.classes)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_classes = get_dataloaders()

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            total += x.size(0)
            correct += (pred == y).sum().item()

        train_acc = correct / total
        train_loss = loss_sum / total

        # валидация
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                total += x.size(0)
                correct += (pred == y).sum().item()
        val_acc = correct / total
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
              f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       f"classifier/aircraft_resnet18_{LEVEL}_best.pth")

if __name__ == "__main__":
    train()
