import sys
import torch
import torch.nn as nn
import numpy as np

# Assume models.py is in the same directory
sys.path.append('.')
from models import (
    PatchTST_Quantum_Sentiment,
    Quantum_Kernel,
    Quantum_Encoder,
    Russian_Sentiment_Model,
)
from transformers import PatchTSTConfig, PatchTSTForPrediction

# ---------- 1. Wrapper for Russian_Sentiment_Model to provide last_hidden_state ----------
class RussianSentimentWithHidden(Russian_Sentiment_Model):
    """Extends Russian_Sentiment_Model to return an object with .last_hidden_state.
    Input shape: (batch, num_articles, seq_len) – token IDs.
    Output: object with .last_hidden_state of shape (batch, num_articles, seq_len, hidden_dim).
    """

    def __init__(self, model_name, num_labels):
        super().__init__()


    def forward(self, input_ids):
        batch_size, num_articles, seq_len = input_ids.shape
        input_ids_flat = input_ids.view(batch_size * num_articles, seq_len)
        attention_mask = torch.ones_like(input_ids_flat)

        # Call transformer directly to get hidden states
        outputs = self.transformer(
            input_ids=input_ids_flat,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = outputs.hidden_states[-1]          # (batch*num_articles, seq_len, hidden_dim)
        hidden_dim = last_hidden.shape[-1]
        last_hidden = last_hidden.view(batch_size, num_articles, seq_len, hidden_dim)

        class Output:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state
        return Output(last_hidden)


# ---------- 2. Model parameters ----------
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

# Sentiment model
sentiment_model = RussianSentimentWithHidden(
    model_name="blanchefort/rubert-base-cased-sentiment",
    num_labels=3
)

actual_hidden_dim = sentiment_model.transformer.config.hidden_size
sentiment_embed_dim = actual_hidden_dim

# Quantum model parameters
quantum_dim = 8
quantum_stride = 2
quantum_depth = 1
dim_post_quantum = d_model
max_quantum_register_size = 5

quantum_encoder = Quantum_Encoder(
    embed_size=quantum_dim,
    encoder_type='Amplitude',
    wires=range(3),
    device='default.qubit',
    pad_val=0,
    out=False
)

quantum_model_config = {
    'layer_type': 'SimplifiedTwoDesign',
    'wires': range(3),
    'layer_config': None,
    'encoder': quantum_encoder,
    'device': 'default.qubit',
    'uncorr_wires': ()
}

# Factory to pass the already instantiated sentiment model to the combined model
def sentiment_factory(config):
    return sentiment_model

# ---------- 3. Instantiate the combined model ----------
combined_model = PatchTST_Quantum_Sentiment(
    time_series_model=PatchTSTForPrediction,
    time_series_config=time_series_config,
    sentiment_model=sentiment_factory,
    sentiment_config={},
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

# ---------- 4. Prepare dummy input ----------
# Time series input (batch, seq_len, num_features)
time_series_inputs = torch.randn(batch_size, seq_len, num_features)

# News input – dummy token IDs (batch, num_articles, seq_len_text)
num_articles = 3
seq_len_text = 128
vocab_size = 30522
news_inputs = torch.randint(0, vocab_size, (batch_size, num_articles, seq_len_text))

# ---------- 5. Forward pass ----------
output = combined_model(time_series_inputs, news_inputs)
print(f"Forward output shape: {output.shape}")   # (batch, prediction_length)

# ---------- 6. Simple backward pass ----------
target = torch.randn(batch_size, prediction_length)
loss = nn.MSELoss()(output, target)
loss.backward()
print("Backward pass completed. Gradients exist:", any(
    p.grad is not None for p in combined_model.parameters()
))
