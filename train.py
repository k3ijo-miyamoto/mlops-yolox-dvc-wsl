import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from model import MyModel

def load_dataset(path):
    df = pd.read_csv(path)
    x = torch.tensor(df[["age"]].values, dtype=torch.float32)
    y = torch.tensor(df[["age"]].values, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=32)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    dataloader = load_dataset(cfg.data_path)
    model = MyModel(cfg.model.input_dim, cfg.model.output_dim, cfg.train.lr)
    trainer = pl.Trainer(max_epochs=cfg.train.max_epochs)
    trainer.fit(model, dataloader)
    torch.save(model.state_dict(), cfg.output_model)

if __name__ == "__main__":
    main()
