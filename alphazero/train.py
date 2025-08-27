import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from play import *
from connectfour import *
from mcts import *
from model import *
from utils import *

# numbner of rounds of play and train to perform
rounds = 10

def train_model(model, data, epochs=10, batch_size=64, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare tensors
    state_tensors = torch.stack([d[0] for d in data])           # shape: [N, 2, 6, 7]
    policy_targets = torch.stack([torch.tensor(d[1]) for d in data])  # shape: [N, 7]
    value_targets = torch.tensor([d[2] for d in data], dtype=torch.float32)  # shape: [N]

    dataset = TensorDataset(state_tensors, policy_targets, value_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for states, target_policies, target_values in dataloader:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            optimizer.zero_grad()
            pred_policies, pred_values = model(states)
            loss = alpha_zero_loss(pred_policies, pred_values, target_policies, target_values)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Init model
    model = AlphaZeroCNN().to(device)
    
    for i in range(rounds):
        # 2. Self-play to generate data
        self_play_data = generate_self_play_data(model, num_games=500, simulations=1000)

        # 3. Train on the data
        train_model(model, self_play_data, epochs=100, batch_size=32, lr=1e-3, device=device)

    torch.save(model.state_dict(), "alphazero_connect4.pth")
