import pytorch_lightning as pl
import torch.nn as nn
import torch

class MyModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, lr):
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
