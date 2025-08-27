import numpy as np
import torch

def encode_board(board, current_player, device='cuda'):
    """
    Encode the board for AlphaZero input.
    
    Args:
        board (np.ndarray): shape (6, 7), with 0 (empty), 1 (player 1), 2 (player 2)
        current_player (int): 1 or 2

    Returns:
        torch.Tensor: shape (2, 6, 7)
    """
    player_board = (board == current_player).astype(np.float32)
    opponent_board = (board != 0) & (board != current_player)
    opponent_board = opponent_board.astype(np.float32)
    
    state_tensor = np.array([player_board, opponent_board])
    return torch.tensor(state_tensor, dtype=torch.float32, device=device)