import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

import text_preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# torch.set_float32_matmul_precision('high')


class PositionEncoding(nn.Module):
    def __init__(self, d_model=2,
                 max_len=6):  # dmodel > dim of embeddings, max_len > max no of tokens our transformer can process (input and output combined)

        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        div_term = 1 / torch.tensor(10000.0) ** (embedding_index / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0), :]


class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__()
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = 0
        self.col_dim = 1

    def forward(self, embeddings_for_q, embeddings_for_k, embeddings_for_v, mask=None):
        q = self.W_q(embeddings_for_q)
        k = self.W_k(embeddings_for_k)
        v = self.W_v(embeddings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim) ** 0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)
        return attention_scores


class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self, n_tokens=4, d_model=2, max_len=6):
        super().__init__()
        self.we = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
        self.pe = PositionEncoding(d_model=d_model, max_len=max_len)

        self.self_attention = Attention(d_model=d_model)

        self.fc_layer = nn.Linear(in_features=d_model, out_features=n_tokens)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0)), device=self.device))
        mask = mask == 0

        self_attention_values = self.self_attention(position_encoded, position_encoded, position_encoded, mask=mask)
        residual_connection_values = position_encoded + self_attention_values
        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",  # Minimize the monitored metric (e.g., loss)
            factor=0.5,  # Reduce the learning rate by this factor
            patience=10,  # Wait 10 epochs with no improvement before reducing
            verbose=True  # Print a message when the learning rate is reduced
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",  # Metric to monitor for scheduler adjustments
            },
        }
    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_folder_path = "C:\\Workspace-ML\\text_data"
    dataloader, vocab_size, tokens_to_idx, idx_to_tokens = text_preprocessing.text_to_data(dataset_folder_path)
    model = DecoderOnlyTransformer(n_tokens=vocab_size, d_model=64, max_len=1024).to(device)
    print(next(model.parameters()).device)

    logger = TensorBoardLogger("tb_logs", name="model-lr-scheduler-long-seq-dataset")
    trainer = L.Trainer(max_epochs=5000, log_every_n_steps=25, accelerator="gpu", devices=1, logger=logger)
    trainer.fit(model, train_dataloaders=dataloader)
