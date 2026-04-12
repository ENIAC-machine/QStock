import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.append('.')
from models import (
    PatchTST_Quantum_Sentiment,
    Quantum_Kernel,
    Quantum_Encoder,
    Sentiment_Model
)

from transformers import PatchTSTConfig, PatchTSTForPrediction

# ----------  Model parameters ----------
batch_size = 4
seq_len = 10               # time series sequence length
num_features = 5
prediction_length = 2
d_model = 8                # hidden dimension for PatchTST

# Time series config (PatchTST)
time_series_config = PatchTSTConfig(
    context_length=seq_len,
    prediction_length=prediction_length,
    d_model=d_model,
    num_input_channels=num_features,
    patch_length=5,
    num_attention_heads=2,
    num_hidden_layers=1,
    output_hidden_states=True
)


#Sentiment model shenanigans
sentiment_embed_dim = 768 
sentiment_config = {
        'model_name' : "blanchefort/rubert-base-cased-sentiment".strip(),
        'num_labels' : 3,
        'device' : 'cpu'
        }


# Quantum model parameters
quantum_dim = 15
quantum_stride = 2
quantum_depth = 1
dim_post_quantum = d_model
max_quantum_register_size = 5

quantum_model_config = {
    'layer_type' : 'StronglyEntanglingLayers',
    'n_layers' : 2,
    'wires': range(4),
    'layer_config': None,
    'encoder': Quantum_Encoder,
    'device': 'default.qubit',
    'uncorr_wires': (),
    'encoder_config' : {'embed_size' : quantum_dim,
                        'encoder_type' : 'Amplitude',
                        'wires' : range(4),
                        'device' : 'default.qubit',
                        'pad_val' : 0,
                        'n_layers' : 2,
                        'out' : False
                        }
    }

# ---------- Instantiate the combined model ----------
combined_model = PatchTST_Quantum_Sentiment(
    time_series_model=PatchTSTForPrediction,
    time_series_config=time_series_config,
    sentiment_model=Sentiment_Model,
    sentiment_config=sentiment_config,
    sentiment_embed_dim=sentiment_embed_dim,
    quantum_model=Quantum_Kernel,
    quantum_model_config=quantum_model_config,
    quantum_dim=quantum_dim,
    quantum_stride=quantum_stride,
    quantum_depth=quantum_depth,
    dim_post_quantum=dim_post_quantum,
    max_quantum_register_size=max_quantum_register_size
)

print("Combined model instantiated successfully.")

# ---------- Prepare dummy input ----------
# Time series input (batch, seq_len, num_features)
time_series_inputs = torch.randn(batch_size, seq_len, num_features)

# News input – dummy token IDs (batch, num_articles, seq_len_text)
num_articles = 3
seq_len_text = 128
vocab_size = 30522
news_inputs = ['this is good', 'this is bad', 'nice', 'really bad']

# ---------- Forward pass ----------
output = combined_model(time_series_inputs, news_inputs)
print(f"Forward output shape: {output.shape}")   # (batch, prediction_length)

# ---------- Simple backward pass ----------
target = torch.randn(batch_size, prediction_length)
loss = nn.MSELoss()(output, target)
loss.backward()
print("Backward pass completed. Gradients exist:", any(
    p.grad is not None for p in combined_model.parameters()
))
