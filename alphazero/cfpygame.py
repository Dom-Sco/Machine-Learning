import pygame
import sys
import numpy as np
import random
import copy
from model import AlphaZeroCNN
from mcts import mcts_search
import torch

# Model Import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AlphaZeroCNN()
model.load_state_dict(torch.load("alphazero_connect4.pth", map_location=device))
model.eval()
model.to(device)

# Constants
ROW_COUNT = 6
COLUMN_COUNT = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
WIDTH = COLUMN_COUNT * SQUARESIZE
HEIGHT = (ROW_COUNT + 1) * SQUARESIZE  # Extra row on top for dropping pieces
SIZE = (WIDTH, HEIGHT)

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Players
PLAYER = 0
AI = 1

# Pieces
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

# This class is used for making game states usable in MCTS 

class ConnectFourState:
    def __init__(self, board=None, player=AI_PIECE):
        self.board = np.copy(board) if board is not None else create_board()
        self.player = player  # AI_PIECE or PLAYER_PIECE

    def get_legal_actions(self):
        return [c for c in range(COLUMN_COUNT) if is_valid_location(self.board, c)]

    def move(self, col):
        # Copy board and apply move
        new_board = np.copy(self.board)
        row = get_next_open_row(new_board, col)
        drop_piece(new_board, row, col, self.player)
        next_player = PLAYER_PIECE if self.player == AI_PIECE else AI_PIECE
        return ConnectFourState(new_board, next_player)

    def is_terminal(self):
        return (
            winning_move(self.board, PLAYER_PIECE) or
            winning_move(self.board, AI_PIECE) or
            len(self.get_legal_actions()) == 0
        )

    def get_result(self):
        # From the perspective of the player who just moved
        opponent = PLAYER_PIECE if self.player == AI_PIECE else AI_PIECE
        if winning_move(self.board, opponent):
            return -1  # You (self.player) lost
        elif winning_move(self.board, self.player):
            return 1   # You won
        else:
            return 0   # Draw or non-terminal

    def __hash__(self):
        return hash(self.board.tostring())

    def __eq__(self, other):
        return np.array_equal(self.board, other.board) and self.player == other.player


def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[0][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT - 1, -1, -1):
        if board[r][col] == 0:
            return r


def print_board(board):
    print(np.flip(board, 0))


def winning_move(board, piece):
    # Horizontal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if all([board[r][c + i] == piece for i in range(4)]):
                return True

    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all([board[r + i][c] == piece for i in range(4)]):
                return True

    # Positive diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if all([board[r + i][c + i] == piece for i in range(4)]):
                return True

    # Negative diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if all([board[r - i][c + i] == piece for i in range(4)]):
                return True

    return False


def draw_board(board, screen):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            # Draw the board background
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, (r + 1) * SQUARESIZE, SQUARESIZE, SQUARESIZE))
            # Draw empty black circles (holes)
            pygame.draw.circle(screen, BLACK, (
                int(c * SQUARESIZE + SQUARESIZE / 2),
                int((r + 1) * SQUARESIZE + SQUARESIZE / 2)
            ), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            piece = board[r][c]
            if piece == PLAYER_PIECE:
                color = RED
            elif piece == AI_PIECE:
                color = YELLOW
            else:
                continue

            # Draw the piece at the correct board position
            pygame.draw.circle(screen, color, (
                int(c * SQUARESIZE + SQUARESIZE / 2),
                int((r + 1) * SQUARESIZE + SQUARESIZE / 2)
            ), RADIUS)

    pygame.display.update()


def animate_drop(screen, board, col, piece_color, final_row):
    for r in range(final_row + 1):
        # Redraw board to clear previous animation frame
        draw_board(board, screen)

        # Draw the falling piece at current row 'r'
        pygame.draw.circle(screen, piece_color, (
            int(col * SQUARESIZE + SQUARESIZE / 2),
            int((r + 1) * SQUARESIZE + SQUARESIZE / 2)
        ), RADIUS)

        pygame.display.update()
        pygame.time.wait(50)  # Delay between steps (50 ms)


def main():
    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("Connect Four - Human vs MCTS AI")
    font = pygame.font.SysFont("monospace", 75)

    board = create_board()
    game_over = False
    turn = random.choice([PLAYER, AI])
    draw_board(board, screen)

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEMOTION and turn == PLAYER:
                screen.fill(BLACK, (0, 0, WIDTH, SQUARESIZE))
                posx = event.pos[0]
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
                pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN and turn == PLAYER:
                screen.fill(BLACK, (0, 0, WIDTH, SQUARESIZE))
                posx = event.pos[0]
                col = int(posx / SQUARESIZE)

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    animate_drop(screen, board, col, RED, row)
                    drop_piece(board, row, col, PLAYER_PIECE)

                    if winning_move(board, PLAYER_PIECE):
                        label = font.render("Player Wins!", True, RED)
                        screen.blit(label, (40, 10))
                        game_over = True

                    draw_board(board, screen)
                    turn = AI
        
        if turn == AI and not game_over:
            current_state = ConnectFourState(board, AI_PIECE)
            col, _ = mcts_search(current_state, model, simulations=10000)
            row = get_next_open_row(board, col)
            pygame.time.wait(500)
            animate_drop(screen, board, col, YELLOW, row)
            drop_piece(board, row, col, AI_PIECE)

            if winning_move(board, AI_PIECE):
                label = font.render("AI Wins!", True, YELLOW)
                screen.blit(label, (40, 10))
                game_over = True

            draw_board(board, screen)
            turn = PLAYER

        

        if game_over:
            pygame.display.update()
            pygame.time.wait(3000)
            pygame.quit()
            sys.exit()


if __name__ == "__main__":
    main()
