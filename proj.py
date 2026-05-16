import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load dataset
data = fetch_california_housing()

X = data.data
y = data.target.reshape(-1, 1)


# preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)


# split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split( X_temp, y_temp, test_size=0.5, random_state=42)


# convert to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# model
class HouseModel(nn.Module):

    def __init__(self, input_size, hidden_layers=[64, 32]):
        super(HouseModel, self).__init__()

        layers = []
        in_features = input_size

        for h in hidden_layers:

            layers.append(nn.Linear(in_features, h))

            layers.append(nn.BatchNorm1d(h))

            layers.append(nn.ReLU())

            layers.append(nn.Dropout(0.2))

            in_features = h

        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# training function
def train_model(hidden_layers, lr=0.001, epochs=200):

    print("\nTraining Model:", hidden_layers)

    model = HouseModel(X_train.shape[1], hidden_layers)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_loss = 999999
    patience = 15
    counter = 0

    for epoch in range(epochs):

        # train
        model.train()

        outputs = model(X_train)

        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # validation
        model.eval()

        with torch.no_grad():

            val_outputs = model(X_val)

            val_loss = loss_function(val_outputs, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # early stopping
        if val_loss.item() < best_loss:

            best_loss = val_loss.item()

            counter = 0

        else:
            counter += 1

        if counter >= patience:

            print("Stopping early at epoch", epoch)

            break

        if epoch % 10 == 0:

            print(
                "Epoch:",
                epoch,
                "Train Loss:",
                round(loss.item(), 4),
                "Val Loss:",
                round(val_loss.item(), 4),
            )

    # final evaluation
    model.eval()

    with torch.no_grad():

        train_mse = loss_function(model(X_train), y_train).item()

        val_mse = loss_function(model(X_val), y_val).item()

        test_mse = loss_function(model(X_test), y_test).item()

    return train_mse, val_mse, test_mse, train_losses, val_losses


# experiments
experiments = [
    {"name": "Exp1", "hidden": [64, 32], "lr": 0.001},
    {"name": "Exp2", "hidden": [128, 64, 32], "lr": 0.001},
    {"name": "Exp3", "hidden": [64, 32], "lr": 0.01},
]


results = []

all_train_losses = []
all_val_losses = []


# run experiments
for exp in experiments:

    train_mse, val_mse, test_mse, train_losses, val_losses = train_model(
        hidden_layers=exp["hidden"], lr=exp["lr"]
    )

    results.append((exp["name"], train_mse, val_mse, test_mse))

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)


# results
print("\nFinal Results\n")

for r in results:

    print(
        r[0],
        "| Train MSE:",
        round(r[1], 4),
        "| Val MSE:",
        round(r[2], 4),
        "| Test MSE:",
        round(r[3], 4),
    )


# visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

experiment_names = [
    "Exp1",
    "Exp2",
    "Exp3"
]

for i, ax in enumerate(axes):

    # train loss
    ax.plot(
        all_train_losses[i],
        label="Train Loss"
    )

    # validation loss
    ax.plot(
        all_val_losses[i],
        linestyle="--",
        label="Validation Loss"
    )

    ax.set_title(experiment_names[i])

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Loss")

    ax.grid(True)

    ax.legend()

plt.tight_layout()

plt.show()
