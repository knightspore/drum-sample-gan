from time import sleep
from models.discriminator_old import AudioClassifier
import torch, wandb, os
import pandas as pd
import torch.nn as nn
from tqdm import tqdm, trange
from utils.create_csv import CreateCSV
from utils.create_dataset import CreateDS
import colored_traceback.always

# Generate Dataset
def dataset_setup(path):
    if not os.path.exists("data/initial_dataset.csv"):
        data = CreateCSV(path)
        data.csv("data/initial_dataset.csv")
    df = pd.read_csv("data/initial_dataset.csv")
    df = df[["path", "class"]]
    print(df.head())
    print(df.tail())
    print(f"Dataset Length: {len(df)}")
    return CreateDS(df)


"""
Main Section
"""

# Config
num_epochs = 5
batch_size = 8
D_lr, G_lr = (
    2e-4,
    3e-4,
)
loss = nn.BCELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Data

dataset = dataset_setup("D:/Documents/Samples/Beat Packs")
num_train = round(len(dataset) * 0.8)
num_val = len(dataset) - num_train
train_ds, valid_ds = torch.utils.data.random_split(dataset, [num_train, num_val])

combo_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True
)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True
)
valid_loader = torch.utils.data.DataLoader(
    valid_ds, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True
)

# Setup Models

D = AudioClassifier()  # Discriminator
D_opt = torch.optim.Adam(D.parameters(), lr=D_lr)
D.to(device)

test = torch.rand(1, 2, 64, 344).to(device)
out = D(test)
print(out.shape)

# G = generator()
# G_opt = torch.optim.Adam(G.parameters(), lr=G_lr)
# G.to(device)

# Training Loop

# with trange(num_epochs, unit="epoch") as tepoch:
#     for epoch in tepoch:
#         for idx, data in enumerate(train_loader):
#             print(idx)
