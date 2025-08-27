import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AlphaZeroCNN(nn.Module):
    def __init__(self, board_height=6, board_width=7, action_size=7, num_res_blocks=5):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.action_size = action_size

        # Initial convolution
        self.conv = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_height * board_width, action_size)

        # --- Value Head ---
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_height * board_width, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x, legal_moves_mask=None):
        # Shared body
        x = F.relu(self.bn(self.conv(x)))       # [B, 64, 6, 7]
        x = self.res_blocks(x)                  # [B, 64, 6, 7]

        # --- Policy Head ---
        p = F.relu(self.policy_bn(self.policy_conv(x)))  # [B, 2, 6, 7]
        p = p.view(p.size(0), -1)                        # [B, 2*6*7]
        p = self.policy_fc(p)                            # [B, 7] (logits)

        if legal_moves_mask is not None:
            p = p.masked_fill(~legal_moves_mask, float('-1e9'))

        log_probs = F.log_softmax(p, dim=1)              # [B, 7]

        # --- Value Head ---
        v = F.relu(self.value_bn(self.value_conv(x)))    # [B, 1, 6, 7]
        v = v.view(v.size(0), -1)                        # [B, 6*7]
        v = F.relu(self.value_fc1(v))                    # [B, 64]
        v = torch.tanh(self.value_fc2(v))                # [B, 1]

        return log_probs, v.squeeze(1)


'''
class AlphaZeroCNN(nn.Module):
    def __init__(self, board_height=6, board_width=7, action_size=7):
        super(AlphaZeroCNN, self).__init__()
        self.board_height = board_height
        self.board_width = board_width
        self.action_size = action_size

        # Shared conv layers
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_height * board_width, action_size)

        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_height * board_width, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x, legal_moves_mask=None):
        # x: [batch, 2, 6, 7]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # --- Policy Head ---
        p = F.relu(self.policy_conv(x))              # [B, 2, 6, 7]
        p = p.view(p.size(0), -1)                    # [B, 2*6*7]
        p = self.policy_fc(p)                        # [B, 7] (logits)

        if legal_moves_mask is not None:
            # Set logits of illegal moves to large negative value
            p = p.masked_fill(~legal_moves_mask, float('-1e9'))

        log_probs = F.log_softmax(p, dim=1)          # [B, 7]

        # --- Value Head ---
        v = F.relu(self.value_conv(x))               # [B, 1, 6, 7]
        v = v.view(v.size(0), -1)                    # [B, 6*7]
        v = F.relu(self.value_fc1(v))                # [B, 64]
        v = torch.tanh(self.value_fc2(v))            # [B, 1]

        return log_probs, v.squeeze(1)
'''


def alpha_zero_loss(pred_policy, pred_value, target_policy, target_value):
    policy_loss = -torch.sum(target_policy * pred_policy, dim=1).mean()
    value_loss = F.mse_loss(pred_value, target_value)
    return policy_loss + value_loss