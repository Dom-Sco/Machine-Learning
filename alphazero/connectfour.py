import numpy as np

# --- Game Constants ---
ROW_COUNT = 6
COLUMN_COUNT = 7
PLAYER_PIECE = 1
AI_PIECE = 2

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)

def is_valid_location(board, col):
    return board[0][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT-1, -1, -1):
        if board[r][col] == 0:
            return r
    return None

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def winning_move(board, piece):
    # Check horizontal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if all(board[r][c+i] == piece for i in range(4)):
                return True
    # Check vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True
    # Check positive diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True
    # Check negative diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if all(board[r-i][c+i] == piece for i in range(4)):
                return True
    return False

class ConnectFourState:
    def __init__(self, board=None, player=AI_PIECE):
        self.board = np.copy(board) if board is not None else self.create_board()
        self.player = player  # AI_PIECE or PLAYER_PIECE

    def get_legal_actions(self):
        return [c for c in range(COLUMN_COUNT) if self.is_valid_location(self.board, c)]

    def move(self, col):
        # Copy board and apply move
        new_board = np.copy(self.board)
        row = self.get_next_open_row(new_board, col)
        self.drop_piece(new_board, row, col, self.player)
        next_player = PLAYER_PIECE if self.player == AI_PIECE else AI_PIECE
        return ConnectFourState(new_board, next_player)

    def is_terminal(self):
        return (
            self.winning_move(self.board, PLAYER_PIECE) or
            self.winning_move(self.board, AI_PIECE) or
            len(self.get_legal_actions()) == 0
        )

    def get_result(self):
        # From the perspective of the player who just moved
        opponent = PLAYER_PIECE if self.player == AI_PIECE else AI_PIECE
        if self.winning_move(self.board, opponent):
            return -1  # You (self.player) lost
        elif self.winning_move(self.board, self.player):
            return 1   # You won
        else:
            return 0   # Draw or non-terminal

    def clone(self):
        return ConnectFourState(self.board.copy(), self.player)

    def get_legal_moves(self):
        return self.get_legal_actions()

    def get_reward(self):
        return self.get_result()

    def make_move(self, col):
        next_state = self.move(col)
        self.board = next_state.board
        self.player = next_state.player

    def __hash__(self):
        return hash(self.board.tobytes())

    def __eq__(self, other):
        return np.array_equal(self.board, other.board) and self.player == other.player

    # --- Static helper methods ---
    @staticmethod
    def create_board():
        return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)

    @staticmethod
    def is_valid_location(board, col):
        return board[0][col] == 0

    @staticmethod
    def get_next_open_row(board, col):
        for r in range(ROW_COUNT - 1, -1, -1):
            if board[r][col] == 0:
                return r
        raise ValueError(f"Column {col} is full")

    @staticmethod
    def drop_piece(board, row, col, piece):
        board[row][col] = piece

    @staticmethod
    def winning_move(board, piece):
        # Check horizontal
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if all([board[r][c+i] == piece for i in range(4)]):
                    return True
        # Check vertical
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if all([board[r+i][c] == piece for i in range(4)]):
                    return True
        # Check positive diagonal
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if all([board[r+i][c+i] == piece for i in range(4)]):
                    return True
        # Check negative diagonal
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if all([board[r-i][c+i] == piece for i in range(4)]):
                    return True
        return False

class ConnectFourGame:
    def __init__(self):
        self.board = create_board()
        self.current_player = PLAYER_PIECE
        self.game_over = False
        self.winner = None

    def reset(self):
        self.board = create_board()
        self.current_player = PLAYER_PIECE
        self.game_over = False
        self.winner = None

    def get_state(self):
        return ConnectFourState(self.board, self.current_player)

    def get_legal_actions(self):
        return [c for c in range(COLUMN_COUNT) if is_valid_location(self.board, c)]

    def step(self, action):
        if not is_valid_location(self.board, action):
            raise ValueError(f"Invalid move: column {action} is full.")

        row = get_next_open_row(self.board, action)
        drop_piece(self.board, row, action, self.current_player)

        if winning_move(self.board, self.current_player):
            self.game_over = True
            self.winner = self.current_player
        elif len(self.get_legal_actions()) == 0:
            self.game_over = True
            self.winner = 0  # draw

        # Switch player
        self.current_player = PLAYER_PIECE if self.current_player == AI_PIECE else AI_PIECE

    def is_terminal(self):
        return self.game_over

    def get_result(self):
        if not self.game_over:
            return None
        if self.winner == 0:
            return 0
        return 1 if self.winner == AI_PIECE else -1
