import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import (Russian_Sentiment_Model,
                    Quantum_Encoder,
                    Quantum_Kernel,
                    PatchTST_Quantum_Sentiment
                    )  # adjust imports as needed

from transformers import PatchTSTConfig

# --- Synthetic data generation ---
def create_synthetic_data(num_samples=5000, num_features=7):
    t = np.arange(num_samples)
    ts_data = np.zeros((num_samples, num_features))
    for i in range(num_features):
        ts_data[:, i] = np.sin(2 * np.pi * t / (i+1) * 10) + 0.2 * np.random.randn(num_samples)
    # Create simple news texts (one per time step)
    news_texts = [f"News {i} with {'positive' if i%2==0 else 'negative'} sentiment" for i in range(num_samples)]
    return ts_data, news_texts

# --- Dataset ---
class HybridDataset(Dataset):
    def __init__(self, ts_data, news_texts, context_len, forecast_horizon):
        self.ts_data = torch.tensor(ts_data, dtype=torch.float32)
        self.news_texts = news_texts
        self.context_len = context_len
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.ts_data) - self.context_len - self.forecast_horizon + 1

    def __getitem__(self, idx):
        past_ts = self.ts_data[idx:idx+self.context_len]
        future_ts = self.ts_data[idx+self.context_len:idx+self.context_len+self.forecast_horizon]
        past_news = self.news_texts[idx:idx+self.context_len]
        return past_ts, future_ts, past_news

# --- Training parameters ---
context_len = 336
forecast_horizon = 96
num_features = 7
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate data
ts_data, news_texts = create_synthetic_data()
dataset = HybridDataset(ts_data, news_texts, context_len, forecast_horizon)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

def collate_fn(batch):
    past_ts = torch.stack([b[0] for b in batch])
    future_ts = torch.stack([b[1] for b in batch])
    past_news = [b[2] for b in batch]
    return past_ts, future_ts, past_news

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# --- Model configuration ---
time_series_config = PatchTSTConfig(
    num_input_channels=num_features,
    context_length=context_len,
    prediction_length=forecast_horizon,
    patch_length=16,
    patch_stride=16,
    d_model=128,
    num_attention_heads=16,
    num_hidden_layers=3,
    dropout=0.2,
    output_hidden_states=True
).to_dict()  # Convert to dict for easier passing

sentiment_config = {"model_name": "blanchefort/rubert-base-cased-sentiment", "num_labels": 3}
quantum_model_config = {
    "layer_type": "SimplifiedTwoDesign",
    "wires": range(5),
    "encoder": Quantum_Encoder(embed_size=8, wires=range(5), out=True)
}

model = PatchTST_Quantum_Sentiment(
    time_series_config=time_series_config,
    sentiment_config=sentiment_config,
    quantum_model_config=quantum_model_config
).to(device)

# --- Training loop ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for past_ts, future_ts, past_news in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        past_ts = past_ts.to(device)
        future_ts = future_ts.to(device)
        # past_news is list of lists; device is not relevant for strings
        optimizer.zero_grad()
        pred = model(past_ts, past_news)
        loss = criterion(pred, future_ts)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * past_ts.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for past_ts, future_ts, past_news in val_loader:
            past_ts = past_ts.to(device)
            future_ts = future_ts.to(device)
            pred = model(past_ts, past_news)
            val_loss += criterion(pred, future_ts).item() * past_ts.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}: Train Loss {train_loss:.6f}, Val Loss {val_loss:.6f}")

# --- Test evaluation ---
model.eval()
test_loss = 0
with torch.no_grad():
    for past_ts, future_ts, past_news in test_loader:
        past_ts = past_ts.to(device)
        future_ts = future_ts.to(device)
        pred = model(past_ts, past_news)
        test_loss += criterion(pred, future_ts).item() * past_ts.size(0)
test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.6f}")
