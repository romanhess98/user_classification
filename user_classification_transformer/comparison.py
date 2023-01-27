import torch
import pytorch_lightning as pl
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import optuna
from sklearn.model_selection import train_test_split

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: %s" % device)

full_aug = transforms.Compose(
    [
        transforms.RandomApply(
            [transforms.RandomAffine(20, (0.05, 0.05), (0.95, 1.05))], 0.5
        ),
        transforms.ToTensor(),
    ]
)

# Init DataLoader from MNIST Dataset
# batch_size = 64
combined_ds = torchvision.datasets.MNIST(
    ".", train=True, download=True, transform=full_aug
)
# combined_loader = torch.utils.data.DataLoader(
#     combined_ds, batch_size=batch_size, num_workers=8
# )
train_idx, val_idx = train_test_split(range(combined_ds.__len__()), test_size=0.2)
train_ds_full = torchvision.datasets.MNIST(
    ".", train=True, download=True, transform=full_aug
)
val_ds_full = torchvision.datasets.MNIST(
    ".", train=True, download=True, transform=transforms.ToTensor()
)
train_ds = torch.utils.data.Subset(train_ds_full, train_idx)
val_ds = torch.utils.data.Subset(val_ds_full, val_idx)
test_ds = torchvision.datasets.MNIST(
    ".", train=False, download=True, transform=transforms.ToTensor()
)
# test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8)


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, model_dim, max_len, trainable=False):
        """
        constructor of sinusoid encoding class

        :param model_dim: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()

        if trainable:
            self.encoding = nn.Parameter(torch.randn(max_len, model_dim))
        else:
            # same size with input matrix (for adding with input matrix)
            encoding = torch.zeros(max_len, model_dim, requires_grad=False)
            self.register_buffer("encoding", encoding)

            pos = torch.arange(0, max_len)
            pos = pos.float().unsqueeze(dim=1)
            # 1D => 2D unsqueeze to represent word's position

            _2i = torch.arange(0, model_dim, step=2).float()
            # 'i' means index of model_dim (e.g. embedding size = 50, 'i' = [0,50])
            # "step=2" means 'i' multiplied with two (same with 2 * i)

            self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / model_dim)))
            self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / model_dim)))
            # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, model_dim = 512]

        seq_len = x.size(1)
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, model_dim = 512]
        # it will add with tok_emb : [128, 30, 512]


class Attention(nn.Module):
    def __init__(self, token_dim, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        # TODO
        self.W_Q = nn.Linear(token_dim, hidden_dim)
        self.W_K = nn.Linear(token_dim, hidden_dim)
        self.W_V = nn.Linear(token_dim, hidden_dim)
        self.sqrt_dim = math.sqrt(hidden_dim)

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # TODO
        out = Q @ K.mT
        out /= self.sqrt_dim
        out = nn.functional.softmax(out, -1)
        return out @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, token_dim, hidden_dim, nr_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.nr_heads = nr_heads
        attention = [Attention(token_dim, hidden_dim) for _ in range(nr_heads)]
        # TODO:
        self.attention = nn.ModuleList(attention)
        self.W_concat = nn.Linear(hidden_dim * nr_heads, token_dim)

    def forward(self, x):
        # TODO:
        cat = torch.cat([att(x) for att in self.attention], 2)
        return self.W_concat(cat)


class LayerNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(model_dim))
        self.beta = nn.Parameter(torch.zeros(model_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out


class FFN(nn.Module):
    def __init__(self, model_dim, hidden_dim, drop_prob):
        super(FFN, self).__init__()
        # TODO
        self.linear1 = nn.Linear(model_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, model_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        # TODO:
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return self.dropout(out)


class Transformer_module(nn.Module):
    def __init__(self, token_dim, hidden_dim, ffn_hidden, nr_heads, drop_prob):
        super(Transformer_module, self).__init__()
        # TODO:
        self.attention = MultiHeadAttention(token_dim, hidden_dim, nr_heads)
        self.norm1 = LayerNorm(token_dim)
        self.ffn = FFN(token_dim, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(token_dim)

    def forward(self, x):
        # TODO:
        # 1. compute self attention
        attention = self.attention(x)
        # 2. add and norm
        norm1 = self.norm1(x + attention)
        # 3. positionwise feed forward network
        ffn = self.ffn(norm1)
        # 4. add and norm
        return self.norm2(norm1 + ffn)


def split_up_patches(x, patch_size):
    # h = x.shape[-2]
    # w = x.shape[-1]
    patches = nn.functional.unfold(x, kernel_size=patch_size, stride=patch_size)
    patches = torch.permute(
        patches, (0, 2, 1)
    )  # note: index convention is (n_batches, n_tokens, hidden_dim)!
    return patches


# test splitting into patches
# x_test_split = torch.randn(32, 1, 28, 28)
# x_test_split_patches = split_up_patches(x_test_split, 4)
# print(x_test_split_patches.shape)


class Transformer(pl.LightningModule):
    def __init__(
        self,
        token_dim=256,
        hidden_dim=64,
        num_class=10,
        nr_layers=3,
        nr_heads=3,
        patch_size=4,
        ffn_hidden=512,
        img_width=28,
        learning_rate=8e-4,
        drop_prob=0.1,
        trainable_pos_emb=True,
        max_num_patch=50,
    ):
        super().__init__()
        self.num_class = num_class
        self.nr_layers = nr_layers
        self.nr_heads = nr_heads
        self.learning_rate = learning_rate
        self.patch_size = patch_size
        self.drop_prob = drop_prob

        self.train_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_class, average="weighted"
        )
        self.val_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_class, average="weighted"
        )
        self.test_acc = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_class, average="weighted"
        )

        # TODO
        self.input = nn.Linear(patch_size * patch_size, token_dim)
        # Transformer blocks
        transformer_modules = [
            Transformer_module(token_dim, hidden_dim, ffn_hidden, nr_heads, drop_prob)
            for _ in range(nr_layers)
        ]
        self.transformer_modules = nn.Sequential(*transformer_modules)
        # classification head
        self.classification = nn.Sequential(
            nn.Linear(token_dim, ffn_hidden),
            LayerNorm(ffn_hidden),
            nn.Tanh(),
            nn.Dropout(drop_prob),
            nn.Linear(ffn_hidden, num_class),
        )
        self.pos_enc = PositionalEncoding(token_dim, max_num_patch, trainable_pos_emb)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.patch_dil = img_width // patch_size
        assert self.patch_dil * patch_size == img_width, "Image size not supported!"

    def forward(self, x):
        # TODO:
        patches = split_up_patches(x, self.patch_size)
        inputs = self.input(patches)
        inputs = torch.cat([self.cls_token.expand(inputs.size()[0], -1, -1), inputs], 1)
        inputs += self.pos_enc(inputs)
        out = self.transformer_modules(inputs)
        # out = torch.mean(out, 1)
        return self.classification(out[:, 0])

    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch

        # Forward pass
        outputs = self(images)

        loss = nn.functional.cross_entropy(outputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        pred_labels = torch.argmax(outputs, 1)
        self.train_acc(pred_labels, labels)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, val_batch, batch_idx):
        images, labels = val_batch
        outputs = self(images)

        loss = nn.functional.cross_entropy(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        pred_labels = torch.argmax(outputs, 1)

        self.val_acc(pred_labels, labels)

        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        pred_labels = torch.argmax(outputs, 1)

        self.test_acc(pred_labels, labels)

        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt


def objective(trial: optuna.Trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [64, 128])
    token_dim = trial.suggest_categorical("token_dim", [128, 256, 512, 768])
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    nr_layers = trial.suggest_int("nr_layers", 1, 12)
    nr_heads = trial.suggest_int("nr_heads", 1, 12)
    ffn_hidden = trial.suggest_categorical("ffn_hidden", [256, 512, 768, 1024])
    drop_prob = trial.suggest_float("drop_prob", 0.0, 0.5)
    trainable_pos_emb = trial.suggest_categorical("trainable_pos_emb", [True, False])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, num_workers=4)

    # early_stopping = pl.callbacks.EarlyStopping(monitor="val_acc", patience=10, mode="max")
    # cktpt = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="val_acc", mode="max")
    prune = optuna.integration.PyTorchLightningPruningCallback(
        trial, monitor="val_loss"
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=100,
        callbacks=[prune],
        num_sanity_val_steps=0,
    )

    model = Transformer(
        token_dim=token_dim,
        hidden_dim=hidden_dim,
        num_class=10,
        nr_layers=nr_layers,
        nr_heads=nr_heads,
        patch_size=4,
        ffn_hidden=ffn_hidden,
        img_width=28,
        learning_rate=learning_rate,
        drop_prob=drop_prob,
        trainable_pos_emb=trainable_pos_emb,
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.logger.log_hyperparams(
        {
            "token_dim": token_dim,
            "hidden_dim": hidden_dim,
            "num_class": 10,
            "nr_layers": nr_layers,
            "nr_heads": nr_heads,
            "patch_size": 4,
            "ffn_hidden": ffn_hidden,
            "img_width": 28,
            "learning_rate": learning_rate,
            "drop_prob": drop_prob,
            "trainable_pos_emb": trainable_pos_emb,
        }
    )

    return trainer.callback_metrics["val_loss"].item()


pruner = optuna.pruners.HyperbandPruner(3, 30, 2)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=100)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

optuna.visualization.plot_optimization_history(study).write_image(
    "optimization_history.png"
)
optuna.visualization.plot_intermediate_values(study).write_image(
    "intermediate_values.png"
)
optuna.visualization.plot_param_importances(study).write_image("param_importances.png")
optuna.visualization.plot_parallel_coordinate(study).write_image(
    "parallel_coordinate.png"
)

"""
Best trial:
  Value: 0.06449469178915024
  Params:
    learning_rate: 0.00027770561450174327
    batch_size: 128
    token_dim: 256
    hidden_dim: 128
    nr_layers: 4
    ffn_hidden: 512
    drop_prob: 0.0
"""
"""
Best is trial 65 with value: 0.03090810403227806 and parameters: 
{
    "learning_rate": 2.914879507683869e-05,
    "token_dim": 768,
    "hidden_dim": 64,
    "nr_layers": 10,
    "nr_heads": 10,
    "ffn_hidden": 768,
    "drop_prob": 0.20122036260938908,
    "trainable_pos_emb": False,
}
"""