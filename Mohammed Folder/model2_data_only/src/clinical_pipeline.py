import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Variables
CSV_PATH = "clinical.csv"
SKIPPED_LOG = "skipped_patients.txt"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoseDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[
            self.df["TCIA Radiomics dummy ID of To_Submit_Final"] != "HNSCC-01-0253"
        ]

        self.targets = [
            "Total prescribed Radiation treatment dose",
            "Radiation treatment_number of fractions",
            "Radiation treatment_dose per fraction",
        ]

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


def plot_and_save(y, ylabel, filename):
    plt.figure()
    plt.plot(range(1, len(y) + 1), y, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} per Epoch")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def save_gradient_direction(model, filename="gradient_direction.png"):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            g = param.grad.view(-1).cpu().detach().numpy()
            grads.extend(g)

    grads = torch.tensor(grads)
    x = torch.arange(len(grads))
    y = torch.zeros_like(x)
    u = torch.ones_like(grads)
    v = grads
    plt.figure()
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)
    plt.title("Gradient Direction (Vector Field)")
    plt.xlabel("Parameter Index")
    plt.ylabel("Gradient Value")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def train():
    dataset = DoseDataset(CSV_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DosePredictor(clinical_input_dim=dataset[0][0].shape[0]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    losses = []
    r2_scores = []
    mses = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        y_true = []
        y_pred = []

        for clinical, targets in loader:
            clinical, targets = clinical.to(DEVICE), targets.to(DEVICE)

            preds = model(clinical)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            y_true.append(targets.detach().cpu())
            y_pred.append(preds.detach().cpu())

        avg_loss = total_loss / len(loader)
        y_true_all = torch.cat(y_true).numpy()
        y_pred_all = torch.cat(y_pred).numpy()
        r2 = r2_score(y_true_all, y_pred_all)

        losses.append(avg_loss)
        r2_scores.append(r2)
        mses.append(avg_loss)

        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f} | R2 Score = {r2:.4f}")

    # Save plots
    plot_and_save(losses, "Loss", "loss.png")
    plot_and_save(r2_scores, "R2 Score", "r2.png")
    plot_and_save(mses, "MSE", "mse.png")

    # Save gradient direction for last batch
    save_gradient_direction(model)

    torch.save(model.state_dict(), "data_model.pt")
    print("Model and plots saved.")


if __name__ == "__main__":
    train()

