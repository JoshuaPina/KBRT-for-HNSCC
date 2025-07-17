import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import os
from PIL import Image

# Variables
CSV_PATH = "clinical.csv"
IMAGE_DIR = "ct_images_avg/"  # path where images are stored as [ID].png or [ID].jpg
SKIPPED_LOG = "skipped_patients.txt"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
class DoseDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[
            self.df["TCIA Radiomics dummy ID of To_Submit_Final"] != "HNSCC-01-0253"
        ]
        self.image_dir = image_dir
        self.transform = transform

        # Target columns
        self.targets = [
            "Total prescribed Radiation treatment dose",
            "Radiation treatment_number of fractions",
            "Radiation treatment_dose per fraction",
        ]

        # Filter patients witout CT
        skipped_ids = []

        def has_image(row):
            img_id = row["TCIA Radiomics dummy ID of To_Submit_Final"]
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                return True
            else:
                skipped_ids.append(img_id)
                return False

        self.df = self.df[self.df.apply(has_image, axis=1)].reset_index(drop=True)

        # Log skipped patients(possibly create test set with)
        with open(SKIPPED_LOG, "w") as log_file:
            log_file.write("\n".join(skipped_ids))

        # Features
        self.df = pd.get_dummies(
            self.df,
            columns=[
                "Gender",
                "Smoking status",
                "HPV Status",
                "Cancer subsite of origin",
            ],
            dummy_na=True,
        )

        # Feature selection (dropping name + targets + nonnumerical values)
        df_features = self.df.drop(
            columns=["TCIA Radiomics dummy ID of To_Submit_Final", *self.targets]
        )

        self.features = df_features.select_dtypes(include=["number"])

        # For feature selection debugging:
        # print(self.features.columns.tolist())
        # exit()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["TCIA Radiomics dummy ID of To_Submit_Final"]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        clinical_vals = pd.to_numeric(row[self.features.columns], errors="coerce")

        # Error checking if NaN
        # print("Problem row:")
        # print(clinical_vals)
        # assert not clinical_vals.isna().any(), f"NaN in clinical values for {img_id}"

        clinical_tensor = torch.tensor(
            clinical_vals.fillna(0).values, dtype=torch.float32
        )

        target_tensor = torch.tensor(
            row[self.targets].astype(float).values, dtype=torch.float32
        )

        return image, clinical_tensor, target_tensor


# Creating the model
class DosePredictor(nn.Module):
    def __init__(self, clinical_input_dim):
        super().__init__()
        self.image_backbone = models.resnet18(pretrained=True)
        self.image_backbone.fc = nn.Identity()

        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_input_dim, 64), nn.ReLU(), nn.Linear(64, 32)
        )

        self.fc = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # output: total dose, #fractions, dose/fraction
        )

    def forward(self, image, clinical):
        x_img = self.image_backbone(image)
        x_clin = self.clinical_net(clinical)
        x = torch.cat([x_img, x_clin], dim=1)
        return self.fc(x)


# Model training loop
def train():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = DoseDataset(CSV_PATH, IMAGE_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DosePredictor(clinical_input_dim=dataset[0][1].shape[0]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, clinical, targets in loader:
            images, clinical, targets = (
                images.to(DEVICE),
                clinical.to(DEVICE),
                targets.to(DEVICE),
            )
            # Error checking if NaN
            # assert not torch.isnan(images).any(), "NaN in image tensor"
            # assert not torch.isnan(clinical).any(), "NaN in clinical tensor"
            # assert not torch.isnan(targets).any(), "NaN in target tensor"

            preds = model(images, clinical)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(preds[0], targets[0])  #See what values we're computing on

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(loader):.4f}")

        # Save model weights
        torch.save(model.state_dict(), "HSNCC_model.pt")


if __name__ == "__main__":
    train()
