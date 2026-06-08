import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    Since Transformers have no built-in sense of sequence order,
    this tells the model which timestep each measurement came from.
    """
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PMUTransformer(nn.Module):
    """
    Transformer model for PMU fault detection and classification.

    Architecture:
      1. Input projection: maps 7 PMU features → d_model dimensions
      2. Positional encoding: adds timestep information
      3. Transformer encoder: learns temporal patterns across timesteps
      4. Global average pooling: collapses sequence → single vector
      5. Classification head: outputs fault type probabilities

    Input shape:  (batch_size, window_size, n_features)
                  e.g. (64, 10, 7)
    Output shape: (batch_size, n_classes)
                  e.g. (64, 6)
    """

    def __init__(
        self,
        n_features=7,       # number of PMU channels (Va,Vb,Vc,Ia,Ib,Ic,Freq)
        n_classes=6,        # fault types: Normal,LG,LL,LLG,LLL,LLLG
        d_model=64,         # internal embedding dimension
        n_heads=4,          # number of attention heads
        n_layers=2,         # number of transformer encoder layers
        d_ff=128,           # feedforward layer dimension
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # 1. Project raw PMU features into d_model space
        self.input_projection = nn.Linear(n_features, d_model)

        # 2. Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True    # input shape: (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 4. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)

        # Project to d_model
        x = self.input_projection(x)        # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding(x)            # (batch, seq_len, d_model)

        # Transformer encoder
        x = self.transformer(x)             # (batch, seq_len, d_model)

        # Global average pooling over time dimension
        x = x.mean(dim=1)                   # (batch, d_model)

        # Classification
        x = self.classifier(x)              # (batch, n_classes)
        return x


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = PMUTransformer(
        n_features=7,
        n_classes=6,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1
    )

    # Simulate one batch: 32 samples, 10 timesteps, 7 features
    dummy_input = torch.randn(32, 10, 7)
    output = model(dummy_input)

    print("=" * 50)
    print("PMU Transformer Model Summary")
    print("=" * 50)
    print(f"Input shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Parameters   : {count_parameters(model):,}")
    print("=" * 50)
    print("Model architecture:")
    print(model)
    print("\nSanity check passed!")