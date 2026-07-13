import random
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the smallest architecture (9 Inputs > 4 Hidden -> 9 Outputs)
class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 4),          # Connects all 9 Inputs to all 4 Hidden nodes
            nn.ReLU(),                                    # Filters the output of those 4 Hidden nodes
            nn.Linear(4,9)            # Connects those 4 Hidden nodes to all 9 Outputs
        )

    def forward(self, x):
        return self.model(x)


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

# 3. Training Setup
model = TicTacToeNet()
# model.parameter() tells Adam which weights and biases it is allowed to modify
# lr = learning rate
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adaptive Moment Estimation
loss_fn = nn.MSELoss()

# Hyperparameters
episodes = 20000
epsilon = 0.3  # Exploration rate (30% random moves to discover strategies)
gamma = 0.9    # Discount factor for future rewards

# Tracking Metrics Variables
p1_wins = 0
p2_wins = 0
ties = 0
window_size = 2000

print ("Training the network via self-play... Please wait.")

for episode in range(1, episodes + 1):
    board = [0] * 9 # Clear board
    turn = 1  # Player 1 starts

    # Track the sequence of states, actions, and rewards for the game
    history = []

    while check_winner(board) is None:
        valid_moves = get_valid_moves(board)

        # Format board state a a tensor for the network
        # We invert the board perspective if it is player -1's turn
        state_tensor = torch.FloatTensor(board) * turn

        # Epsilon-Greedy action selection
        # Decide whether to make a random move or pick the current known best move
        if random.random() < epsilon:
            action = random.choice(valid_moves)
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
            # Mask invalid moves by giving them a very low score
            for i in range(9):
                if i not in valid_moves:
                    q_values[i] = -999.0
            action = torch.argmax(q_values).item()

        # Record state and action taken
        history.append({'state': state_tensor, 'action': action, 'turn': turn})

        # Apply move
        board[action] = turn
        turn = -turn # Switch turn

    # Game over -> Calculate rewards and train
    winner = check_winner(board)

    # Record metrics for the game
    if winner == 1:
        p1_wins += 1
    elif winner == -1:
        p2_wins += 1
    else:
        ties += 1

    # Print out progress every 2,000 episodes
    if episode % window_size == 0:
        p1_pct = (p1_wins / window_size) * 100
        p2_pct = (p2_wins / window_size) * 100
        tie_pct = (ties / window_size) * 100

        print(f"Games {episode - window_size + 1:5d} to {episode:5d} | "
              f"P1 Wins: {p1_pct:4.1f}% | "
              f"P2 Wins: {p2_pct:4.1f}% | "
              f"Ties: {tie_pct:4.1f}%")

        # Reset counters for the next window block
        p1_wins, p2_wins, ties = 0, 0, 0

    # Update the network backward through the game history
    # Backpropagation training loop
    for entry in reversed(history):
        p_turn = entry['turn']

        # Reward calculation from this player's perspective
        if winner == 0:
            reward = 0.5   # Tie is a good outcome
        elif winner == p_turn:
            reward = 1.0   # Won
        else:
            reward = -1.0  # Lost

        # Target Q-Value calculation (Bellman Equation)
        # How to reward a move made early in the game that directly leads to a victory five turns later.
        target = reward

        # Update network weights
        current_q = model(entry['state'])
        target_q = current_q.clone().detach()
        target_q[entry['action']] = target

        loss = loss_fn(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("\nTraining complete!")

# Save the mmodel's weights and architecture state
model_path = "small_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model successfully saved to {model_path}!")


# Interactive Human vs AI Play
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

