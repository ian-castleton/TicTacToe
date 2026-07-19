import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import deque
import random as rnd

replay_buffer = deque(maxlen=25000)
batch_size = 64


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


def evaluate_against_random(model, games=500, agent_plays="X"):
    wins, losses, draws = 0, 0, 0
    agent_side = 1 if agent_plays == "X" else -1

    for _ in range(games):
        env = TicTacToeEnv()
        state = env.reset()
        done = False
        while not done:
            legal = env.get_legal_moves()
            if env.current_player == agent_side:
                state_t = torch.FloatTensor(state)
                with torch.no_grad():
                    q = model(state_t).clone()
                for i in range(9):
                    if i not in legal:
                        q[i] = float('-inf')
                action = torch.argmax(q).item()
            else:
                action = random.choice(legal)
            state, reward, done, status = env.step(action)

        winner = env.check_winner()
        if winner == agent_side:
            wins += 1
        elif winner == -agent_side:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws


# -------------------------------------------------------------
# 3. SELF-PLAY TRAINING ALGORITHM
# -------------------------------------------------------------
def train_reinforcement_learning(episodes=20000):
    model = TicTacNet(hidden_size=32)

    target_model = TicTacNet(hidden_size=32)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # lowered from 0.005
    criterion = nn.MSELoss()

    env = TicTacToeEnv()
    epsilon = 0.4
    decay = 0.9999
    gamma = 0.9

    window_size = 5000
    window_stats = {"win_x": 0, "win_o": 0, "draw": 0}
    window_losses = []

    print(f"Training network via self-play for {episodes} episodes...")

    start = time.time()
    for episode in range(episodes):
        state = env.reset()
        done = False
        game_history = []

        # ---- 1. PLAY A GAME (unchanged from your version) ----
        while not done:
            state_t = torch.FloatTensor(state)
            legal_moves = env.get_legal_moves()

            if random.random() < epsilon:
                action = random.choice(legal_moves)
            else:
                with torch.no_grad():
                    q_values = model(state_t).clone()
                    for i in range(9):
                        if i not in legal_moves:
                            q_values[i] = float('-inf')
                    action = torch.argmax(q_values).item()

            next_state, reward, done, status = env.step(action)
            game_history.append((state_t, action, reward))

            if done:
                # If game ended in a win, the opponent lost. Retroactively penalize opponent's last move.
                if status == "win" and len(game_history) > 1:
                    prev_state, prev_action, prev_reward = game_history[-2]
                    game_history[-2] = (prev_state, prev_action, -1.0)
            else:
                state = next_state

        # ---- 2. PUSH TRANSITIONS INTO THE REPLAY BUFFER ----
        # (No training happens here — just recording what happened,
        # same idea as game_history before, but stored persistently
        # across many episodes instead of thrown away after one.)
        n = len(game_history)
        for i, (s_t, act, r) in enumerate(game_history):
            if i < n - 2:
                # Normal case: bootstrap later from your own next turn
                next_s_t = game_history[i + 2][0]
                replay_buffer.append((s_t, act, r, next_s_t, False))
            elif i == n - 2 and status == "draw":
                # Second-to-last move in a draw: outcome is fully known
                # already (0.9 * 0.5), so store it as a fixed terminal target
                fixed_target = r + gamma * game_history[i + 1][2]
                replay_buffer.append((s_t, act, fixed_target, None, True))
            else:
                # Final move of the game (any outcome), or the
                # second-to-last move of a win/loss (already fixed to -1.0/1.0)
                replay_buffer.append((s_t, act, r, None, True))

        # ---- 3. TRAIN ON A RANDOM MINI-BATCH FROM THE BUFFER ----
        if len(replay_buffer) >= batch_size:
            batch = rnd.sample(replay_buffer, batch_size)
            for s_t, act, r, next_s_t, is_terminal in batch:
                target = r
                if not is_terminal:
                    with torch.no_grad():
                        next_legal = [j for j in range(9) if next_s_t[j].item() == 0]

                        # Double DQN: select with live model, evaluate with target model
                        next_q_live = model(next_s_t).clone()
                        for j in range(9):
                            if j not in next_legal:
                                next_q_live[j] = float('-inf')
                        best_action = torch.argmax(next_q_live).item()

                        next_q_target = target_model(next_s_t)
                        target += gamma * next_q_target[best_action].item()

                current_q = model(s_t)[act]
                loss = criterion(current_q, torch.tensor(target, dtype=torch.float32))
                window_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # ---- 4. BOOKKEEPING (unchanged) ----
        winner = env.check_winner()
        if winner == 1:
            window_stats["win_x"] += 1
        elif winner == -1:
            window_stats["win_o"] += 1
        else:
            window_stats["draw"] += 1

        epsilon = max(0.05, epsilon * decay)

        if (episode + 1) % 500 == 0:
            target_model.load_state_dict(model.state_dict())

        if (episode + 1) % window_size == 0:
            total = sum(window_stats.values())
            avg_loss = sum(window_losses) / len(window_losses) if window_losses else 0.0

            print(f"Episode {episode + 1}/{episodes} | Epsilon: {epsilon:.3f} | Avg Loss: {avg_loss:.4f}")
            print(f"  Last {total} games — X wins: {window_stats['win_x'] / total * 100:.1f}% | "
                  f"O wins: {window_stats['win_o'] / total * 100:.1f}% | "
                  f"Draws: {window_stats['draw'] / total * 100:.1f}%")

            wx, lx, dx = evaluate_against_random(model, games=200, agent_plays="X")
            wo, lo, do = evaluate_against_random(model, games=200, agent_plays="O")
            print(f"  Eval — X: Win {wx / 2:.0f}% Draw {dx / 2:.0f}% | O: Win {wo / 2:.0f}% Draw {do / 2:.0f}%")

            end = time.time()
            print(f"  Time taken: {(end - start):.1f}s")
            start = end

            window_stats = {"win_x": 0, "win_o": 0, "draw": 0}
            window_losses = []

    print("Training finished!\n")
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

