import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(x)


class DeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 160),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(160, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class MSEClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, outputs, labels):
        probs = torch.softmax(outputs, dim=1)
        targets = torch.nn.functional.one_hot(labels, num_classes=10).float()
        return self.loss(probs, targets)


DATASET_CONFIGS = {
    "mnist": {
        "class": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
        "label": "MNIST"
    },
    "fashionmnist": {
        "class": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "label": "Fashion-MNIST"
    },
    "kmnist": {
        "class": datasets.KMNIST,
        "mean": (0.1918,),
        "std": (0.3483,),
        "label": "KMNIST"
    }
}


def get_positive_int_env(name):
    value = os.environ.get(name)
    if not value:
        return None

    try:
        parsed = int(value)
    except ValueError:
        return None

    return parsed if parsed > 0 else None


def get_dataset_limit(name, render_default):
    configured_size = get_positive_int_env(name)
    if configured_size is not None:
        return configured_size

    if os.environ.get("RENDER") == "true" or os.environ.get("RAILWAY_ENVIRONMENT"):
        return render_default

    return None


def limited_subset(dataset, size):
    if size is None or size >= len(dataset):
        return dataset
    return Subset(dataset, range(size))


def get_model(name):
    name = name.lower()

    if name == "tinymlp":
        return TinyMLP()
    elif name == "simplecnn":
        return SimpleCNN()
    elif name == "deepercnn":
        return DeeperCNN()
    else:
        raise ValueError(f"Unsupported model architecture: {name}")


def get_dataloaders(batch_size=64, dataset_name="mnist"):
    dataset_key = dataset_name.lower()
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    config = DATASET_CONFIGS[dataset_key]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"])
    ])

    train_dataset = config["class"](
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = config["class"](
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    configured_train_size = get_dataset_limit("TRAIN_SUBSET_SIZE", 256)
    configured_val_size = get_dataset_limit("VAL_SUBSET_SIZE", 64)
    configured_test_size = get_dataset_limit("TEST_SUBSET_SIZE", 64)

    if configured_train_size or configured_val_size:
        train_size = configured_train_size or 50000
        val_size = configured_val_size or max(1, len(train_dataset) - train_size)
        requested_size = train_size + val_size

        if requested_size > len(train_dataset):
            scale = len(train_dataset) / requested_size
            train_size = max(1, int(train_size * scale))
            val_size = max(1, len(train_dataset) - train_size)

        train_dataset = limited_subset(train_dataset, train_size + val_size)
    else:
        train_size = 50000
        val_size = len(train_dataset) - train_size

    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_dataset = limited_subset(test_dataset, configured_test_size)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_optimizer(name, model_params, lr):
    name = name.lower()

    if name == "sgd":
        return optim.SGD(model_params, lr=lr)
    elif name == "momentum":
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    elif name == "rmsprop":
        return optim.RMSprop(model_params, lr=lr)
    elif name == "adam":
        return optim.Adam(model_params, lr=lr)
    elif name == "adamw":
        return optim.AdamW(model_params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def get_criterion(name):
    name = name.lower()

    if name == "crossentropy":
        return nn.CrossEntropyLoss()
    elif name == "labelsmoothing":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    elif name == "mse":
        return MSEClassificationLoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    optimizer_name="adam",
    lr=0.001,
    epochs=3,
    batch_size=64,
    dataset_name="mnist",
    model_name="simplecnn",
    loss_name="crossentropy",
    progress_callback=None
):
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        dataset_name=dataset_name
    )

    model = get_model(model_name).to(device)
    criterion = get_criterion(loss_name)
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    epoch_times = []  # Per-epoch timing in seconds

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_time = round(time.time() - epoch_start, 3)

        train_losses.append(round(epoch_loss, 4))
        train_accuracies.append(round(epoch_acc * 100, 2))
        epoch_times.append(epoch_time)

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        val_losses.append(round(val_loss, 4))
        val_accuracies.append(round(val_acc * 100, 2))

        elapsed_time = round(time.time() - start_time, 2)
        epoch_payload = {
            "current_epoch": epoch + 1,
            "total_epochs": epochs,
            "progress": round(((epoch + 1) / epochs) * 100, 2),
            "elapsed_time_seconds": elapsed_time,
            "train_loss": round(epoch_loss, 4),
            "train_accuracy": round(epoch_acc * 100, 2),
            "validation_loss": round(val_loss, 4),
            "validation_accuracy": round(val_acc * 100, 2),
            "epoch_time_seconds": epoch_time
        }

        if progress_callback:
            progress_callback(epoch_payload)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}% - "
            f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc*100:.2f}% - "
            f"Time: {epoch_time:.2f}s"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    total_time = round(time.time() - start_time, 2)

    return {
        "optimizer": optimizer_name,
        "dataset": dataset_name,
        "model": model_name,
        "loss_function": loss_name,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "validation_losses": val_losses,
        "validation_accuracies": val_accuracies,
        "epoch_times": epoch_times,
        "final_test_loss": round(test_loss, 4),
        "final_test_accuracy": round(test_acc * 100, 2),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_train_accuracy": train_accuracies[-1] if train_accuracies else None,
        "final_validation_loss": val_losses[-1] if val_losses else None,
        "final_validation_accuracy": val_accuracies[-1] if val_accuracies else None,
        "training_time_seconds": total_time
    }


if __name__ == "__main__":
    result = train_model(
        optimizer_name="adam",
        lr=0.001,
        epochs=2,
        batch_size=64
    )
    print(result)
