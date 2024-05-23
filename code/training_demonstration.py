"""
Demonstrates training with smooth check loss.

To do:
- Pull common operations outside of the main routine
- Break symmetry with other DGPs
"""

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import quantile_dnn as qdnn

# Create a class for a DNN with ReLU activation, dropout layers, and Kaiming initialization
class DeepNeuralNetworkReLU(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, dropout_rate):
        super(DeepNeuralNetworkReLU, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            linear = nn.Linear(prev_size, hidden_size)
            nn.init.kaiming_normal_(linear.weight, mode="fan_in")
            self.layers.append(linear)
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        self.layers.append(nn.Linear(prev_size, output_dim))

    # Define the forward pass
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    """
    Default training demonstration.
    """

    # 1. DGP
    torch.manual_seed(12345)  # Set the seed for reproducibility

    N = 15000  # observation count
    K = 1  # feature count

    # Draw independent features from a uniform distribution and multiply by 5
    dnn_feature = torch.rand(N, K)

    # Idiosyncratic error under quantile DGP
    U = torch.rand(N, 1)

    # Calculate outcomes
    outcomes = torch.mul(dnn_feature, U)

    # Plot outcomes against features
    plt.scatter(dnn_feature, outcomes, s = 0.5)
    plt.xlabel("$X$")
    plt.ylabel("$Y$")
    plt.savefig("../out/training_demonstration/scatter.png", bbox_inches="tight")
    plt.clf()
    
    # 2. Common hyperparameters
    num_epochs = 1000
    hidden_sizes = [64, 64]
    output_dim = outcomes.shape[1]
    dropout_rate = 0.0
    learning_rate = 1e-3
    weight_decay = 0.0

    # Plotting objects
    dnn_feature_plot = np.linspace(0, 1, 101)
    dnn_feature_plot_tensor = torch.tensor(dnn_feature_plot).reshape(-1, 1).float()

    # 3. MSE training
    model = DeepNeuralNetworkReLU(
        input_dim=K,
        hidden_sizes=hidden_sizes,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    )

    # Use MSE loss
    loss_fn = nn.MSELoss()

    # Initialize the optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Store the loss history
    loss_history = []

    for epoch in tqdm(range(num_epochs)):
        # Forward pass
        outcomes_pred = model(dnn_feature)

        # Calculate the loss
        loss = loss_fn(outcomes_pred, outcomes)
        if np.isnan(float(loss)):
            print("NaN loss detected :(")
            print(loss)
            break

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

        # Store loss
        loss_history.append(loss.item())

    outcomes_pred = model(dnn_feature)

    # Loss history
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("../out/training_demonstration/loss_mse.png", bbox_inches="tight")
    plt.clf()

    # Obtain model predictions on plotting feature tensor
    output_plot_tensor = model(dnn_feature_plot_tensor)
    output_plot = output_plot_tensor.detach().numpy()

    # Expected value: E[Y|X=x] = 0.5x
    E = dnn_feature_plot * 0.5

    # Plot
    plt.plot(dnn_feature_plot, output_plot, label = "Estimate")
    plt.plot(dnn_feature_plot, E, linestyle="--", label = "Estimand")
    plt.xlabel("x")
    plt.ylabel("$\mathbb{E}[Y|X=x]$")
    plt.legend()
    plt.savefig("../out/training_demonstration/pred_mse.png", bbox_inches="tight")
    plt.clf()

    # 4. Training with smooth check loss
    quantiles = [0.25, 0.5, 0.75]

    for tau in quantiles:
        print("Quantile:", tau)

        # Initialize the model
        model = DeepNeuralNetworkReLU(
            input_dim=K,
            hidden_sizes=hidden_sizes,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
        )

        # Use smooth check loss
        alpha = 0.1

        loss_fn = qdnn.SmoothCheckLoss(tau, alpha, reduction="mean")

        loss_fn_check = qdnn.CheckLoss(tau, reduction="mean")

        # Initialize the optimizer
        optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Store the loss history
        loss_history = []
        loss_check_history = []

        for epoch in tqdm(range(num_epochs)):
            # Forward pass
            outcomes_pred = model(dnn_feature)

            # Calculate the loss
            loss = loss_fn(outcomes_pred, outcomes)

            if np.isnan(float(loss)):
                    print("NaN loss detected :(")
                    print(loss)
                    break

            # Calculate check loss
            with torch.no_grad():
                loss_check = loss_fn_check(outcomes_pred, outcomes)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update parameters
            optimizer.step()

            # Store loss
            loss_history.append(loss.item())
            loss_check_history.append(loss_check.item())

        outcomes_pred = model(dnn_feature)

        plt.plot(loss_history, label="Smooth Check Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss, $\\tau = {tau}$")
        plt.legend()
        plt.savefig(f"../out/training_demonstration/loss_smooth_q{tau * 100:.0f}.png", bbox_inches="tight")
        plt.clf()

        plt.plot(loss_check_history, label="Check Loss", linestyle="--", color="tab:orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss, $\\tau = {tau}$")
        plt.savefig(f"../out/training_demonstration/loss_check_q{tau * 100:.0f}.png", bbox_inches="tight")
        plt.clf()
        
        plt.plot([loss / loss_history[0] for loss in loss_history], label="Smooth Check Loss")
        plt.plot([loss_check / loss_check_history[0] for loss_check in loss_check_history], label="Check Loss", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss, $\\tau = {tau}$")
        plt.legend()
        plt.savefig(f"../out/training_demonstration/loss_compare_q{tau * 100:.0f}.png", bbox_inches="tight")
        plt.clf()

        # Obtain model predictions on plotting feature tensor
        output_plot_tensor = model(dnn_feature_plot_tensor)
        output_plot = output_plot_tensor.detach().numpy()

        # Conditional quantile: Q_(u)[Y|X=x] = u * x
        Q = dnn_feature_plot * tau

        # Plot
        plt.plot(dnn_feature_plot, output_plot, label = "Estimate")
        plt.plot(dnn_feature_plot, Q, linestyle="--", label = "Estimand")
        plt.xlabel("x")
        plt.ylabel(f"$\mathbb{{Q}}_{{{tau}}}[Y|X=x]$")
        plt.title(f"Quantile Regression, $\\tau = {tau}$")
        plt.legend()
        plt.savefig(f"../out/training_demonstration/pred_q{tau * 100:.0f}.png", bbox_inches="tight")
        plt.clf()