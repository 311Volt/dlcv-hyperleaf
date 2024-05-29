import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import sys
import time
import timeit
from PIL import Image
import numpy as np
import tifffile

DSDIR = None

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DLCVNet(nn.Module):
    def __init__(self):
        super(DLCVNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(204, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # Move to GPU before dummy input
        self.conv_layers.to(device)

        # Dummy input to determine the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 204, 48, 352).to(device)
            flattened_size = self.conv_layers(dummy_input).view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24, 8),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def map_grain_weight(raw_weight):
    return np.log2(raw_weight) - 10

def unmap_grain_weight(mapped_weight):
    return np.exp2(mapped_weight + 10)

def path_to_image_id(img_id):
    img_id_str = "{:05d}".format(int(img_id))
    return os.path.join(DSDIR, "images", img_id_str + ".tiff")

def read_image(img_id):
    imgpath = path_to_image_id(img_id)
    return (np.transpose(
        tifffile.imread(imgpath),
        axes=(2, 0, 1)
    ) / 65535.0).astype(np.float32)

def map_dataset_row(row: np.array):
    t0 = timeit.default_timer()
    image_id = int(row[0])
    raw_grain_weight = float(row[1])
    mapped_grain_weight = map_grain_weight(raw_grain_weight)
    stomatal_conductance = float(row[2])
    phi_ps2 = float(row[3])
    fertilizer = float(row[4])
    prob_heerup = float(row[5])
    prob_kvium = float(row[6])
    prob_rembrandt = float(row[7])
    prob_sheriff = float(row[8])

    img = read_image(image_id)
    result = np.array([
        mapped_grain_weight, 
        stomatal_conductance, phi_ps2, fertilizer, 
        prob_heerup, prob_kvium, prob_rembrandt, prob_sheriff
    ], dtype=np.float32)
    t1 = timeit.default_timer()

    return img, result

def prepare_submission(out_fn, sample_submission: pd.DataFrame, model):
    output_data = sample_submission.copy(deep=True)
    model.eval()
    with torch.no_grad():
        for idx, row in output_data.iterrows():
            input_image = torch.tensor([read_image(row['ImageId'])]).to(device)
            if input_image.shape[1] != 204:
                continue
            input_image = input_image.permute(0, 3, 1, 2)  # Convert to NCHW format
            pred = model(input_image)[0].cpu().numpy()
            output_data.at[idx, 'GrainWeight'] = unmap_grain_weight(float(pred[0]))
            output_data.at[idx, 'Gsw'] = float(pred[1])
            output_data.at[idx, 'PhiPS2'] = float(pred[2])
            output_data.at[idx, 'Fertilizer'] = float(pred[3])
            output_data.at[idx, 'Heerup'] = float(pred[4])
            output_data.at[idx, 'Kvium'] = float(pred[5])
            output_data.at[idx, 'Rembrandt'] = float(pred[6])
            output_data.at[idx, 'Sheriff'] = float(pred[7])
    output_data.to_csv(out_fn)

def entry(in_mdl=None):
    if in_mdl is not None:
        model = torch.load(in_mdl).to(device)
        prepare_submission("submission.csv", pd.read_csv(os.path.join(DSDIR, "sample_submission.csv")), model)
        exit()

    train_csv = pd.read_csv(os.path.join(DSDIR, "train.csv"))
    train_meta = train_csv.iloc[:1440]
    val_meta = train_csv.iloc[1440:]

    batch_size = 16

    train_data = []
    val_data = []

    for _, row in train_meta.iterrows():
        img, label = map_dataset_row(row.to_numpy())
        train_data.append((torch.tensor(img), torch.tensor(label)))

    for _, row in val_meta.iterrows():
        img, label = map_dataset_row(row.to_numpy())
        val_data.append((torch.tensor(img), torch.tensor(label)))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = DLCVNet().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)

    num_epochs = 600

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 2, 3, 1)  # Ensure this permutation matches the expected [N, C, H, W] format
            inputs = inputs.clone().detach().float()  # Convert dtype without creating a new tensor
            labels = labels.clone().detach().float()  # Convert dtype without creating a new tensor

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.permute(0, 2, 3, 1)  # Ensure this permutation matches the expected [N, C, H, W] format
                inputs = inputs.clone().detach().float()  # Convert dtype without creating a new tensor
                labels = labels.clone().detach().float()  # Convert dtype without creating a new tensor

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}')

        if (epoch + 1) % 50 == 0:
            torch.save(model, f'saved-model-epoch{epoch + 1}-{val_loss:.10f}.pt')

    torch.save(model, "final.pt")

if __name__ == "__main__":
    DSDIR = "./dataset_hyperleaf/"
    print(sys.argv)
    if len(sys.argv) > 1:
        entry("dlcvnet.pt")

    entry()
