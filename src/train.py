import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0
from dataloader import get_dataloaders
import mlflow
import yaml

def train_model():
    config = yaml.safe_load(open("configs/config.yaml"))
    data_path = config["dataset"]["path"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]

    train_loader, test_loader = get_dataloaders(data_path, batch_size)

    mlflow.start_run()

    model = efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(1280, 10)

    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch}: Loss = {avg_loss}")

    torch.save(model.state_dict(), "models/model.pth")
    mlflow.log_artifact("models/model.pth")

    mlflow.end_run()

if __name__ == "__main__":
    train_model()
