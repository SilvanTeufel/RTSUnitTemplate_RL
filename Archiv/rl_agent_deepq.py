import mmap
import struct
import time
import json
import win32event
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque  # For the replay buffer

# Define shared memory name (same as Unreal)
SHARED_MEMORY_NAME = "Global\\UnrealRLSharedMemory"
MEMORY_SIZE = 4098    # Should match Unreal's size

# Structure format (must match Unreal's `SharedData`)
SHARED_MEMORY_FORMAT = "??" + "2048s" + "2048s"


# --- RL Agent Parameters ---
LEARNING_RATE = 0.001  # Adjusted learning rate
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000
NUM_EPISODES = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 100
REPLAY_BUFFER_SIZE = 10000
# --- RL Agent Parameters ---

# Define the state space based on your GameStateData (including resources now)
STATE_SPACE_SIZE = 12  # MyUnits, EnemyUnits, MyHealth, EnemyHealth, MyAttack, EnemyAttack, AgentPosX, AgentPosY, AgentPosZ, AvgFriendlyPosX, AvgFriendlyPosY, AvgFriendlyPosZ, PrimaryResource, SecondaryResource, TertiaryResource, RareResource, EpicResource, LegendaryResource
# Define the action space
# Define the action space
ACTION_SPACE = [
    # 0-1: Set InputActionValue
    {"type": "Control", "input_value": 0.0, "alt": False, "ctrl": False, "action": "none", "camera_state": 0},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "none", "camera_state": 0},

    # 2-3: Set AltIsPressed
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "alt_select", "camera_state": 0},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "no_modifier", "camera_state": 0}, # No Alt pressed

    # 4-5: Set CtrlIsPressed (when Alt is False)
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "ctrl_select", "camera_state": 0},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "no_modifier", "camera_state": 0}, # No Ctrl pressed

    # Combined Actions (Example - you'll need to define more based on your 10 points)
    # Example for Alt + SwitchControllerStateMachine - Select Units with Tag Alt1-6
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 21},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 22},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 23},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 24},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 25},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 26},

    # Example for Ctrl + SwitchControllerStateMachine - Select Units with Tag Strg1-6, F1-F4 And Q, W, E ,R
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 18}, # R
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 9}, # Q
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 10}, # E
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 1}, # W

    # Select Units with Tag Strg1-6
     {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 21},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 22},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 23},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 24},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 25},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 26},

    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 27},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 28},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 29},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 30},

    # Example for No Modifier + SwitchControllerStateMachine (Ability Use) - Use Ability 1 - 6 From Array with Current Index
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 21},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 22},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 23},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 24},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 25},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 26},

    # Example for Ctrl + Change Ability Index
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "change_ability_index", "camera_state": 13},

    # Example for No Modifier + Move Camera +/- x,y
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "move_camera", "camera_state": 1},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "move_camera", "camera_state": 2},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "move_camera", "camera_state": 3},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "move_camera", "camera_state": 4},

    # Example for No Modifier + Stop Move Camera +/- x,y
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "stop_move_camera", "camera_state": 111},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "stop_move_camera", "camera_state": 222},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "stop_move_camera", "camera_state": 333},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "stop_move_camera", "camera_state": 444},

    # Example for No Modifier + Left Click
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "left_click", "camera_state": 1}, # Using a camera state to trigger
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "left_click", "camera_state": 2}, # Using a different camera state to trigger

    # Example for No Modifier + Right Click
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "right_click", "camera_state": 1}, # Reusing a camera state for example
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "right_click", "camera_state": 2}, # Reusing a camera state for example
]

ACTION_SPACE_SIZE = len(ACTION_SPACE)
ACTIONS = [action_dict["type"] for action_dict in ACTION_SPACE] # Extract action types

class SimpleQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(SimpleQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# Initialize the Q-networks
q_network = SimpleQNetwork(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
target_network = SimpleQNetwork(STATE_SPACE_SIZE, ACTION_SPACE_SIZE)
target_network.load_state_dict(q_network.state_dict()) # Initialize target network with the same weights
target_network.eval()
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return torch.stack(states), \
               torch.tensor(actions, dtype=torch.int64).unsqueeze(1), \
               torch.tensor(rewards, dtype=torch.float32).unsqueeze(1), \
               torch.stack(next_states), \
               torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.buffer)

# Initialize replay buffer
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

def state_to_tensor(game_state_dict):
    """Converts the game state dictionary to a PyTorch tensor."""
    state = [
        game_state_dict.get("MyUnits", 0),
        game_state_dict.get("EnemyUnits", 0),
        game_state_dict.get("MyHealth", 0.0),
        game_state_dict.get("EnemyHealth", 0.0),
        game_state_dict.get("MyAttack", 0.0),
        game_state_dict.get("EnemyAttack", 0.0),
        game_state_dict.get("AgentPosition", [0.0, 0.0, 0.0])[0], # X
        game_state_dict.get("AgentPosition", [0.0, 0.0, 0.0])[1], # Y
        game_state_dict.get("AgentPosition", [0.0, 0.0, 0.0])[2], # Z
        game_state_dict.get("AvgFriendlyPos", [0.0, 0.0, 0.0])[0], # X
        game_state_dict.get("AvgFriendlyPos", [0.0, 0.0, 0.0])[1], # Y
        game_state_dict.get("AvgFriendlyPos", [0.0, 0.0, 0.0])[2], # Z
        game_state_dict.get("AvgEnemyPos", [0.0, 0.0, 0.0])[0], # X
        game_state_dict.get("AvgEnemyPos", [0.0, 0.0, 0.0])[1], # Y
        game_state_dict.get("AvgEnemyPos", [0.0, 0.0, 0.0])[2], # Z
        game_state_dict.get("PrimaryResource", 0.0),
        game_state_dict.get("SecondaryResource", 0.0),
        game_state_dict.get("TertiaryResource", 0.0),
        game_state_dict.get("RareResource", 0.0),
        game_state_dict.get("EpicResource", 0.0),
        game_state_dict.get("LegendaryResource", 0.0),
    ]
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Unsqueeze to add batch dimension

def select_action(state, episode):
    """Selects an action using an epsilon-greedy policy."""
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
              torch.exp(torch.tensor(-1. * episode / EPSILON_DECAY))
    if random.random() > epsilon.item():
        with torch.no_grad():
            return q_network(state).max(1)[1].item()
    else:
        return random.randrange(ACTION_SPACE_SIZE)

def write_action_to_shared_memory(shared_memory, action_dict):
    """Writes an action dictionary (as JSON) into shared memory."""
    action_json_str = json.dumps(action_dict)
    action_bytes = action_json_str.encode("utf-16")[:512]  # Convert to UTF-16
    action_bytes += b"\x00" * (512 - len(action_bytes))  # Pad to 256 wchar_t

    # Create a new shared data block with action
    shared_data = struct.pack(SHARED_MEMORY_FORMAT, False, True, b"\x00" * 512, action_bytes)

    shared_memory.seek(0)
    shared_memory.write(shared_data)
    shared_memory.flush()

def read_shared_memory(shared_memory):
    """Reads game state from shared memory."""
    shared_memory.seek(0)
    data = shared_memory.read(struct.calcsize(SHARED_MEMORY_FORMAT))
    bNewGameStateAvailable, bNewActionAvailable, game_state_bytes, action_bytes = struct.unpack(SHARED_MEMORY_FORMAT, data)

    # Decode the received game state
    game_state = game_state_bytes.decode("utf-16", errors="ignore").strip("\x00")
    action = action_bytes.decode("utf-16", errors="ignore").strip("\x00") # Read the action sent by Unreal (for reward)
    return bNewGameStateAvailable, game_state, action

def extract_reward(previous_game_state, current_game_state, unreal_action_str):
    reward = 0

    # Example rewards based on changes in game state
    if previous_game_state: # Make sure it's not the first state
        # Reward for increasing friendly units
        if current_game_state.get("MyUnits", 0) > previous_game_state.get("MyUnits", 0):
            reward += 0.1

        # Reward for decreasing enemy units
        if current_game_state.get("EnemyUnits", 0) < previous_game_state.get("EnemyUnits", 0):
            reward += 0.2

        # Reward for increasing friendly health
        health_diff = current_game_state.get("MyHealth", 0.0) - previous_game_state.get("MyHealth", 0.0)
        if health_diff > 0:
            reward += health_diff * 0.01

        # Reward for decreasing enemy health
        enemy_health_diff = previous_game_state.get("EnemyHealth", 0.0) - current_game_state.get("EnemyHealth", 0.0)
        if enemy_health_diff > 0:
            reward += enemy_health_diff * 0.02

        # Reward for collecting resources (example for PrimaryResource)
        if current_game_state.get("PrimaryResource", 0.0) > previous_game_state.get("PrimaryResource", 0.0):
            reward += 0.05

        # Penalties for negative changes
        if current_game_state.get("MyUnits", 0) < previous_game_state.get("MyUnits", 0):
            reward -= 0.3

        if current_game_state.get("EnemyUnits", 0) > previous_game_state.get("EnemyUnits", 0):
            reward -= 0.1

        health_diff = current_game_state.get("MyHealth", 0.0) - previous_game_state.get("MyHealth", 0.0)
        if health_diff < 0:
            reward += health_diff * 0.01 # Negative reward

    # Small negative reward for each step to encourage faster completion (optional)
    reward -= 0.001

    return reward

def optimize_model():
    if len(replay_buffer) < BATCH_SIZE:
        return

    transitions = replay_buffer.sample(BATCH_SIZE)
    if transitions is None:
        return
    states, actions, rewards, next_states, dones = transitions

    # Compute Q(s, a) - the model computes Q(s, a) for all actions, then we select the actions taken.
    q_values = q_network(states).gather(1, actions)

    # Compute V(s') for all next states.
    next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
    # Compute the expected Q values
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    # Compute Huber loss
    loss = criterion(q_values, expected_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in q_network.parameters():
        param.grad.data.clamp_(-1, 1) # Clip gradients for stability
    optimizer.step()

if __name__ == "__main__":
    print("[RL Agent] Connecting to shared memory...")

    try:
        shared_memory = mmap.mmap(-1, MEMORY_SIZE, tagname=SHARED_MEMORY_NAME, access=mmap.ACCESS_WRITE)
    except mmap.error as e:
        print(f"Error creating or accessing shared memory: {e}")
        print("Make sure the Unreal Engine application has created the shared memory with the same name.")
        raise

    print("[RL Agent] Ready. Waiting for game state...")

    episode_number = 0
    last_state = None
    last_action_index = None

    while episode_number < NUM_EPISODES:
            time.sleep(0.1)

            bNewGameStateAvailable, game_state_str, unreal_action_str = read_shared_memory(shared_memory)

            if bNewGameStateAvailable and game_state_str:

                game_state_str = game_state_str.strip() # Added this line

                try:
                    game_state_dict = json.loads(game_state_str)
                    current_state_tensor = state_to_tensor(game_state_dict)

                    # Get reward from the previous step (if any)
                    if last_state is not None and last_action_index is not None:
                        reward = extract_reward(last_game_state_dict, game_state_dict, unreal_action_str) # Pass both states
                        done = False # Add your logic to determine if the episode is done
                        replay_buffer.push(last_state, last_action_index, reward, current_state_tensor, done)
                        optimize_model()

                    # Select action using the RL agent
                    action_index = select_action(current_state_tensor, episode_number)
                    chosen_action = ACTION_SPACE[action_index]

                    write_action_to_shared_memory(shared_memory, chosen_action)

                    # Update state and action for the next step
                    last_state = current_state_tensor
                    last_action_index = action_index
                    last_game_state_dict = game_state_dict # Store the current game state

                    # ... (target network update and episode increment) ...

                except json.JSONDecodeError:
                    print("[RL Agent] Error decoding JSON.")
                except Exception as e:
                    print(f"[RL Agent] An error occurred: {e}")
            else:
                time.sleep(0.01)

    print("[RL Agent] Training finished.")