import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

# Wczytanie danych
data = torch.load("../data/imdb_dataset_prepared_embedings.pt")
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]
embeddings = data["embeddings"]

embedding_dim = embeddings.shape[0]
vocab_size = 12849

# Model
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings):
        super(CNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(pretrained_embeddings.T, requires_grad=True)

        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=8)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)

# Model instancja
model = CNNTextClassifier(vocab_size, embedding_dim, embeddings)

# Dataset i DataLoader
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# Loss i optymalizator
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# Trenowanie
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss, total_acc, num_batches = 0.0, 0.0, 0

    start_time = time.time()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.float())
        loss.backward()
        optimizer.step()

        acc = ((preds > 0.5) == (yb > 0.5)).float().mean().item()
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1

    train_loss = total_loss / num_batches
    train_acc = total_acc / num_batches

    model.eval()
    with torch.no_grad():
        preds_test = model(X_test).squeeze()
        test_loss = criterion(preds_test, y_test.float()).item()
        test_acc = ((preds_test > 0.5) == (y_test > 0.5)).float().mean().item()

    elapsed = time.time() - start_time
    print(f"Epoch: {epoch+1} ({elapsed:.2f}s) \tTrain: (l: {train_loss:.4f}, a: {train_acc:.4f}) \tTest: (l: {test_loss:.4f}, a: {test_acc:.4f})")
