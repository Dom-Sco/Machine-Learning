import math
import numpy as np
import random
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from connectfour import *

class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = {}
        self.N = 0  # Visit count
        self.W = 0  # Total value
        self.Q = 0  # Mean value
        self.P = None  # Prior probability from NN

    def is_leaf(self):
        return len(self.children) == 0

def mcts_search(root_state, model, simulations=100, c_puct=1.0):
    root = MCTSNode(root_state)

    # Initial NN prediction
    input_tensor = encode_board(root_state.board, root_state.player).unsqueeze(0)
    logits, _ = model(input_tensor)
    logits = logits.squeeze()
    legal_moves = root_state.get_legal_actions()

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[legal_moves] = True
    masked_logits = logits.masked_fill(~mask, float('-1e9'))
    policy = F.softmax(masked_logits, dim=0).detach().cpu().numpy()

    root.P = policy

    for _ in range(simulations):
        node = root
        path = []

        # Selection
        while not node.is_leaf():
            best_score = -float("inf")
            for a, child in node.children.items():
                u = c_puct * (child.P if child.P is not None else 1.0) * math.sqrt(node.N) / (1 + child.N)
                score = child.Q + u
                if score > best_score:
                    best_score = score
                    best_move = a
                    best_child = child
            node = best_child
            path.append(node)

        # Expansion
        if not node.game.is_terminal():
            legal_moves = node.game.get_legal_actions()
            for move in legal_moves:
                next_game = node.game.move(move)
                child = MCTSNode(next_game, node, move)
                node.children[move] = child

            expand_move = random.choice(legal_moves)
            leaf_node = node.children[expand_move]

            input_tensor = encode_board(leaf_node.game.board, leaf_node.game.player).unsqueeze(0)
            logits, value = model(input_tensor)
            logits = logits.squeeze()

            leaf_legal = leaf_node.game.get_legal_actions()
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[leaf_legal] = True
            probs = F.softmax(logits.masked_fill(~mask, float('-1e9')), dim=0).detach().cpu().numpy()

            for move in leaf_legal:
                next_game = leaf_node.game.move(move)
                child_node = MCTSNode(next_game, leaf_node, move)
                child_node.P = probs[move]
                leaf_node.children[move] = child_node

            v = value.item()
        else:
            v = node.game.get_result()

        # Backpropagation
        for n in path:
            n.N += 1
            n.W += v
            n.Q = n.W / n.N
            v = -v

    # Build π from visit counts
    visit_counts = np.zeros(COLUMN_COUNT)
    for move, child in root.children.items():
        visit_counts[move] = child.N

    pi = visit_counts / (np.sum(visit_counts) + 1e-8)
    best_move = np.argmax(visit_counts)

    return best_move, pi

