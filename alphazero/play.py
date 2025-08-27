from connectfour import *
from mcts import *

def generate_self_play_data(model, num_games=10, simulations=50):
    data = []

    for game_idx in range(num_games):
        game = ConnectFourGame()
        game_history = []  # Store (state, pi) per move

        while not game.is_terminal():
            state = game.get_state()
            legal_actions = state.get_legal_actions()

            # Run MCTS to get move and policy distribution
            move, pi = mcts_search(state, model, simulations=simulations)

            # Encode the state and store it with the policy
            input_tensor = encode_board(state.board, state.player)
            game_history.append((input_tensor, pi, state.player))

            # Apply the chosen move
            game.step(move)

        # Game is over, assign outcome
        final_result = game.get_result()  # 1, -1, or 0
        for input_tensor, pi, player in game_history:
            z = final_result if player == AI_PIECE else -final_result
            data.append((input_tensor, pi, z))

        print(f"Game {game_idx+1}/{num_games} finished. Result: {final_result}")

    return data
