import torch
import torch.nn as nn

# 1. You must define the axact same architecture first
class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 4),          # Connects all 9 Inputs to all 4 Hidden nodes
            nn.ReLU(),                # Filters the otput of those 4 Hidden nodes
            nn.Linear(4,9)            # Connects those 4 Hidden nodes to all 9 Outputs
        )

    def forward(self, x):
        return self.model(x)

# 2. Instantiate the empty network
model = TicTacToeNet()

# 3. Load the saved weights into the network
model.load_state_dict(torch.load("small_model.pth"))

# 4. Set the model to evaluation mode (turns off training mechanics)
model.eval()

print("Trained model loaded successfully and ready to play!")

# 4. Play a game against trained AI
print ("\n--- Play against the AI! (You are X, AI is 0) ---")
board = [0] * 9


def print_board(b):
    symbols = {0: " ", 1: "X", -1: "0"}
    print(f" {symbols[b[0]]} | {symbols[b[1]]} | {symbols[b[2]]} ")
    print("-----------")
    print(f" {symbols[b[3]]} | {symbols[b[4]]} | {symbols[b[5]]} ")
    print("-----------")
    print(f" {symbols[b[6]]} | {symbols[b[7]]} | {symbols[b[8]]} \n")

# 2. Game Environment Helpers
def check_winner(board):
    # All possible winning lines on tic tac toe board
    lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0,3,6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    for line in lines:
        if board[line[0]] == board[line[1]] == board[line[2]] != 0:
            return board[line[0]]  # returns 1 or -1
    if 0 not in board: return 0    # Tie
    return None  # Game ongoing

# Gets list of all the index numbers (0 to 8) that are still empty and available to play on
def get_valid_moves(board):
    return [i for i, v in enumerate(board) if v == 0]


while check_winner(board) is None:
    print_board(board)

    # Human turn (X)
    valid = get_valid_moves(board)
    move = -1
    while move not in valid:
        try:
            move = int(input(f"Enter your move (0-8) Choose from {valid}: "))
        except ValueError:
            pass
    board[move] = 1

    if check_winner(board) is not None:
        break

    # AI Turn (0)
    state_tensor = torch.FloatTensor(board) * -1  # AI views board from its perspective
    with torch.no_grad():
        q_values = model(state_tensor)
    valid_moves = get_valid_moves(board)
    for i in range(9):
        if i not in valid_moves:
            q_values[i] = -999.0
    ai_move = torch.argmax(q_values).item()
    board[ai_move] = -1
    print(f"AI chose square {ai_move}")

print_board(board)
w = check_winner(board)
if w == 1:
    print("You win!")
elif w == -1:
    print("AI wins!")
else:
    print("It's a tie!")

