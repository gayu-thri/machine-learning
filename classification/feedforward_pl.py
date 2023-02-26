#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
input_size = 784  # 28*28 (flattened image)
hidden_size = 500  # number of neurons
num_classes = 10  # Target feature - digits - 0,1,2,3,...9
num_epochs = 5  # Total number of times the model gets to see the whole data
batch_size = 100  # How many samples do we want to pass at a time to the model
learning_rate = 0.001


# Fully connected neural network with one hidden layer
class NeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.save_hyperparameters()
        # output layer only has 10 neurons (10 digits)
        self.test_accuracy = Accuracy()

    # feed forward flow defined here
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)  # activation req. for non-linearity
        out = self.fc2(out)
        return nn.functional.log_softmax(out, dim=1)

    # Optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer

    # Training model
    def training_step(self, train_batch, batch_idx):
        images, labels = train_batch
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(
            device
        )  # -1 because dependent on batch size
        labels = labels.to(device)

        # Forward pass
        outputs = self.forward(images)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)

        tensorboard_logs = {"train_loss": loss}
        # self.log('train_loss', loss)
        return {"loss": loss, "log": tensorboard_logs}

    def test_step(self, test_batch, batch_idx):
        # correct = 0
        # total = 0
        images, labels = test_batch
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        loss = F.cross_entropy(outputs, labels)

        self.test_accuracy.update(predicted, labels)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

        # tensorboard_logs = {'test_loss': loss}
        # return {"test_loss": loss, "log": tensorboard_logs}

    def prepare_data(self) -> None:
        train_dataset = torchvision.datasets.MNIST(
            root="data/mnist",
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )

        test_dataset = torchvision.datasets.MNIST(
            root="data/mnist", train=False, transform=transforms.ToTensor()
        )
        self.train_dataset, self.test_dataset = train_dataset, test_dataset

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        return loader

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=True
        )
        return loader

# uses hyper params from the checkpoint
model = NeuralNet(input_size, hidden_size, num_classes)
# .load_from_checkpoint(
#     "/home/local/ZOHOCORP/gayathri-12052/learnings/pytorch/classification/lightning_logs/MNISTModel/version_0/checkpoints/epoch=5-step=3599.ckpt"
# )

from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("lightning_logs/", name="MNISTModel")
# from pytorch_lightning.loggers import WandbLogger

# wandb_logger = WandbLogger(project='MNIST', # group runs in "MNIST" project
#                            log_model='all') # log all new checkpoints during training

trainer = pl.Trainer(
    max_epochs=num_epochs + 1,
    logger=logger,
    callbacks=[TQDMProgressBar(refresh_rate=1), ModelCheckpoint(dirpath="weights/")],
    # ckpt_path="/home/local/ZOHOCORP/gayathri-12052/learnings/pytorch/classification/lightning_logs/MNISTModel/version_0/checkpoints/epoch=5-step=3599.ckpt",
)
trainer.fit(model)

trainer.test()

for p in model.parameters():
    print(p.shape)

print(model.state_dict())

# dataset = torchvision.datasets.MNIST(root='data/mnist',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)

# dataloader =  torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# print(len(dataloader))
# # 600 * 100 (batch_size)

# for imgs, labels in dataloader:
#     # print(img.shape) #torch.Size([100, 1, 28, 28])
#     print(imgs.reshape(-1, 28*28).shape)
#     break
