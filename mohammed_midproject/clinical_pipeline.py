import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

# Variables
CSV_PATH = "clinical.csv"
SKIPPED_LOG = "skipped_patients.txt"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
class DoseDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[
            self.df["TCIA Radiomics dummy ID of To_Submit_Final"] != "HNSCC-01-0253"
        ]

        # Target columns
        self.targets = [
            "Total prescribed Radiation treatment dose",
            "Radiation treatment_number of fractions",
            "Radiation treatment_dose per fraction",
        ]

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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clinical_vals = pd.to_numeric(row[self.features.columns], errors="coerce")
        clinical_tensor = torch.tensor(
            clinical_vals.fillna(0).values, dtype=torch.float32
        )
        target_tensor = torch.tensor(
            row[self.targets].astype(float).values, dtype=torch.float32
        )
        return clinical_tensor, target_tensor


# Creating the model
class DosePredictor(nn.Module):
    def __init__(self, clinical_input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(clinical_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, clinical):
        return self.model(clinical)


# Model training loop
def train():
    dataset = DoseDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DosePredictor(clinical_input_dim=dataset[0][0].shape[0]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for clinical, targets in loader:
            clinical, targets = clinical.to(DEVICE), targets.to(DEVICE)

            preds = model(clinical)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(loader):.4f}")

    # Save model weights
    torch.save(model.state_dict(), "data_model.pt")
    print("Model saved to data_model.pt")


if __name__ == "__main__":
    train()

