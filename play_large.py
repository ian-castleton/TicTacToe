import torch
import torch.nn as nn
import numpy as np

# -------------------------------------------------------------
# 1. NEURAL NETWORK ARCHITECTURE
# -------------------------------------------------------------

# 1. You must define the axact same architecture first
class TicTacNet(nn.Module):
    def __init__(self, hidden_size=32):  # Boosted from 4 to 32 for perfect mastery
        super(TicTacNet, self).__init__()
        self.fc1 = nn.Linear(9, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 9)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



# -------------------------------------------------------------
# 2. TIC-TAC-TOE ENVIRONMENT
# -------------------------------------------------------------
class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # 0 = empty, 1 = Player X (Goes first), -1 = Player O
        self.board = np.zeros(9, dtype=np.float32)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        # Return state from the perspective of the current player
        return self.board * self.current_player

    def get_legal_moves(self):
        return [i for i, val in enumerate(self.board) if val == 0]

    def check_winner(self):
        win_states = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]  # diagonals
        ]
        for line in win_states:
            b = self.board[line]
            if np.all(b == 1): return 1
            if np.all(b == -1): return -1
        if 0 not in self.board:
            return 0  # Draw
        return None  # Game ongoing

    def step(self, action):
        # Invalid move punishment
        if self.board[action] != 0:
            return self.get_state(), -10, True, "invalid"

        self.board[action] = self.current_player
        winner = self.check_winner()

        if winner is not None:
            if winner == 0:
                return self.get_state(), 0.5, True, "draw"
            else:
                # The player who just moved won
                return self.get_state(), 1.0, True, "win"

        # Switch player turn
        self.current_player = -self.current_player
        return self.get_state(), 0.0, False, "ongoing"



# -------------------------------------------------------------
# 4. INTERACTIVE PLAY AGAINST HUMAN
# -------------------------------------------------------------
def play_human_vs_ai(model):
    env = TicTacToeEnv()

    print("\n=== PLAY AGAINST THE NEURAL NETWORK ===")
    print("Board spots are indexed 0-8 sequentially:")
    print("0 | 1 | 2\n---------\n3 | 4 | 5\n---------\n6 | 7 | 8\n")

    human_piece = input("Do you want to be X (goes first) or O (goes second)? Enter X or O: ").strip().upper()
    while human_piece not in ['X', 'O']:
        human_piece = input("Invalid choice. Choose X or O: ").strip().upper()

    state = env.reset()
    done = False

    # FIX: Explicitly assign which value matches the human player
    # Env starts with env.current_player = 1 (which represents X, going first)
    human_turn_value = 1 if human_piece == 'X' else -1

    while not done:
        # Display current board representation
        display_board = []
        for val in env.board:
            if val == 1:
                display_board.append('X')
            elif val == -1:
                display_board.append('O')
            else:
                display_board.append(' ')

        print(f"\n{display_board[0]} | {display_board[1]} | {display_board[2]}")
        print("---------")
        print(f"{display_board[3]} | {display_board[4]} | {display_board[5]}")
        print("---------")
        print(f"{display_board[6]} | {display_board[7]} | {display_board[8]}\n")

        # FIX: Check if the environment's current player matches the human's value
        if env.current_player == human_turn_value:
            # Human turn logic
            legal = env.get_legal_moves()
            move = -1
            while move not in legal:
                try:
                    move = int(input(f"Your turn ({human_piece}). Enter move (0-8): "))
                    if move not in legal:
                        print("That spot is already occupied or invalid!")
                except ValueError:
                    print("Please enter a clean integer between 0 and 8.")

            state, _, done, status = env.step(move)
        else:
            # AI turn logic
            print("AI is deciding...")
            state_t = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = model(state_t).numpy()

            # Mask out illegal moves so AI doesn't pick them
            legal_moves = env.get_legal_moves()
            for i in range(9):
                if i not in legal_moves:
                    q_values[i] = -float('inf')

            ai_move = np.argmax(q_values)
            ai_piece = 'O' if human_piece == 'X' else 'X'
            print(f"AI ({ai_piece}) plays position: {ai_move}")
            state, _, done, status = env.step(ai_move)

    # Final board state print so the user can see the winning layout
    display_board = ['X' if v == 1 else 'O' if v == -1 else ' ' for v in env.board]
    print(f"\n{display_board[0]} | {display_board[1]} | {display_board[2]}")
    print("---------")
    print(f"{display_board[3]} | {display_board[4]} | {display_board[5]}")
    print("---------")
    print(f"{display_board[6]} | {display_board[7]} | {display_board[8]}\n")

    # Game Over Output Configuration
    winner = env.check_winner()
    if winner == 0:
        print("It's a draw game!")
    elif (winner == 1 and human_piece == 'X') or (winner == -1 and human_piece == 'O'):
        print("Incredible! You defeated the neural network!")
    else:
        print("The AI wins! Better luck next time.")

if __name__ == "__main__":

    # 2. Instantiate the empty network
    trained_model = TicTacNet()

    # 3. Load the saved weights into the network
    trained_model.load_state_dict(torch.load("large_model.pth"))

    # 4. Set the model to evaluation mode (turns off training mechanics)
    trained_model.eval()

    # Start loop to play against it
    while True:
        play_human_vs_ai(trained_model)
        if input("\nPlay again? (y/n): ").lower() != 'y':
            break


