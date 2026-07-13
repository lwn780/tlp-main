import torch
import torch.nn as nn

class TransformerUncertaintyModule(nn.Module):
    def __init__(self, fea_size=22, step_size=25, attention_head=4):
        super().__init__()
        self.fea_size = fea_size
        self.step_size = step_size

        # Encoder
        hidden_dim = [64, 128, 256, 256]
        self.encoder = nn.Sequential(
            nn.Linear(fea_size, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.ReLU(),
        )

        # Transformer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim[-1],
            dim_feedforward=256,
            nhead=attention_head
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=2
        )

        # Two heads: score and log variance
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.var_head = nn.Sequential(
            nn.Linear(hidden_dim[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()   # 保证方差非负
        )

    def forward(self, x):
        # x: [batch, step, fea]
        x = x[:, :self.step_size, :self.fea_size]
        h = self.encoder(x)
        h = h.transpose(0, 1)  # Transformer expects [seq, batch, fea]
        h = self.transformer(h)
        h = h.sum(0)            # sum over steps
        score = self.score_head(h).squeeze(-1)
        log_var = self.var_head(h).squeeze(-1)
        return score, log_var

# ---------------- Test GPU -----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerUncertaintyModule().to(device)

    # 假数据: batch=16, step_size=25, fea_size=22
    x = torch.randn(16, 25, 22).to(device)
    score, log_var = model(x)
    print("score:", score.shape)
    print("log variance:", log_var.shape)
    