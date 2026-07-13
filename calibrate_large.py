import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# -------------------------------------------------------------
# 1. NEURAL NETWORK ARCHITECTURE
# -------------------------------------------------------------
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
# 3. SELF-PLAY TRAINING ALGORITHM
# -------------------------------------------------------------
def train_reinforcement_learning(episodes=20000):
    # Instantiate network, loss, and optimizer
    model = TicTacNet(hidden_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    criterion = nn.MSELoss()

    env = TicTacToeEnv()
    epsilon = 0.4  # Exploration rate
    decay = 0.9999

    print(f"Training network via self-play for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        done = False

        # Track history within the current game to retroactively apply rewards
        game_history = []

        while not done:
            state_t = torch.FloatTensor(state)

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(range(9))
            else:
                with torch.no_grad():
                    q_values = model(state_t)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, status = env.step(action)

            # Record historical step data
            game_history.append((state_t, action, reward))

            if done:
                # If game ended in a win, the opponent lost. Retroactively penalize opponent's last move.
                if status == "win" and len(game_history) > 1:
                    prev_state, prev_action, prev_reward = game_history[-2]
                    game_history[-2] = (prev_state, prev_action, -1.0)
            else:
                state = next_state

        # Batch gradient update at the end of the game
        for i, (s_t, act, r) in enumerate(game_history):
            target = r
            # If the move wasn't terminal, add discounted future predicted reward
            if r != -10 and i < len(game_history) - 2:
                next_s_t = game_history[i + 2][0]  # Look ahead to your next turn's state
                with torch.no_grad():
                    target += 0.9 * torch.max(model(next_s_t)).item()

            current_q = model(s_t)[act]
            loss = criterion(current_q, torch.tensor(target, dtype=torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epsilon = max(0.05, epsilon * decay)

        if (episode + 1) % 5000 == 0:
            print(f"Episode {episode + 1}/{episodes} complete. Exploration (Epsilon): {epsilon:.3f}")

    print("Training finished!\n")

    # Save the mmodel's weights and architecture state
    model_path = "large_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model successfully saved to {model_path}!")

    return model





if __name__ == "__main__":
    # Train the agent
    trained_model = train_reinforcement_learning(episodes=25000)
    # Start loop to play against it
    # while True:
    #     play_human_vs_ai(trained_model)
    #     if input("\nPlay again? (y/n): ").lower() != 'y':
    #         break

