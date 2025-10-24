import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data(json_path="C:/Users/Kapitan Nemo/Documents/μFlux_KM3/data_cnn/imdb_dataset_prepared.json"):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    X_train_raw = data['X_train']
    y_train_raw = data['y_train']
    X_test_raw = data['X_test']
    y_test_raw = data['y_test']
    embeddings_raw = data['embeddings']
    vocab_raw = data['vocab']

    embeddings_np = np.array(embeddings_raw, dtype=np.float32)

    X_train_np_raw = np.array(X_train_raw, dtype=np.int64)
    X_test_np_raw = np.array(X_test_raw, dtype=np.int64)

    X_train_np_adjusted = X_train_np_raw - 1
    X_test_np_adjusted = X_test_np_raw - 1

    X_train = torch.tensor(X_train_np_adjusted, dtype=torch.long)
    y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test_np_adjusted, dtype=torch.long)
    y_test = torch.tensor(y_test_raw, dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_test, y_test, embeddings_np, vocab_raw

class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. Model Definition
class CNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings,
                 num_filters=8, kernel_size_conv=3, kernel_size_pool=8,
                 dense_in_features=128):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        self.conv = nn.Conv1d(in_channels=embedding_dim,
                              out_channels=num_filters,
                              kernel_size=kernel_size_conv)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=kernel_size_pool)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dense_in_features, 1)
        nn.init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        permuted = embedded.permute(0, 2, 1)
        conved = self.conv(permuted)
        activated = self.relu(conved)
        pooled = self.pool(activated)
        flat = self.flatten(pooled)
        logits = self.fc(flat)
        probs = self.sigmoid(logits)
        return probs


def train():
    X_train, y_train, X_test, y_test, embeddings_np, vocab = load_data("C:/Users/Kapitan Nemo/Documents/μFlux_KM3/data_cnn/imdb_dataset_prepared.json")

    embedding_dim = embeddings_np.shape[1]
    vocab_size = embeddings_np.shape[0]

    model = CNN(vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    pretrained_embeddings=embeddings_np,
                    num_filters=8,
                    kernel_size_conv=3,
                    kernel_size_pool=8,
                    dense_in_features=128
                    ).to(device)

    print("\nModel Architecture:")
    print(model)

    batch_size = 64
    train_dataset = Dataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_test_dev = X_test.to(device)
    y_test_dev = y_test.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    def accuracy_fn(y_pred, y_true):
        predicted_classes = (y_pred > 0.5).float()
        correct = (predicted_classes == y_true).float()
        acc = correct.sum() / len(correct)
        return acc

    epochs = 5
    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        start_time = time.time()

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            acc = accuracy_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()


        end_time = time.time()
        epoch_duration = end_time - start_time
        train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_acc / len(train_loader)

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test_dev)
            test_loss = criterion(y_test_pred, y_test_dev).item()
            test_acc = accuracy_fn(y_test_pred, y_test_dev).item()

        print(f"Epoch: {epoch} ({epoch_duration:.2f}s) \t"
              f"Train: (l: {train_loss:.4f}, a: {train_acc:.4f}) \t"
              f"Test: (l: {test_loss:.4f}, a: {test_acc:.4f})")


train_start_time = time.time()
train()
train_end_time = time.time()
print(f"\nTotal training time: {train_end_time - train_start_time:.2f}s")