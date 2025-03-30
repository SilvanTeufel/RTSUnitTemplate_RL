import mmap
import struct
import time
import json
import torch
import torch.nn as nn

# Define shared memory name (same as Unreal)
SHARED_MEMORY_NAME = "Global\\UnrealRLSharedMemory"
MEMORY_SIZE = 4098    # Should match Unreal's size

# Structure format (must match Unreal's `SharedData`)
SHARED_MEMORY_FORMAT = "??" + "2048s" + "2048s"

# Define the state space size (must match the training script)
STATE_SPACE_SIZE = 21

# Define the action space (must match the training script)
ACTION_SPACE = [
    {"type": "Control", "input_value": 0.0, "alt": False, "ctrl": False, "action": "none", "camera_state": 0},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "none", "camera_state": 0},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "alt_select", "camera_state": 0},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "no_modifier", "camera_state": 0},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "ctrl_select", "camera_state": 0},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "no_modifier", "camera_state": 0},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 21},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 22},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 23},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 24},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 25},
    {"type": "Control", "input_value": 1.0, "alt": True, "ctrl": False, "action": "switch_camera_state", "camera_state": 26},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 18},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 9},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "switch_camera_state", "camera_state": 10},
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
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 21},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 22},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 23},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 24},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 25},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "switch_camera_state_ability", "camera_state": 26},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": True, "action": "change_ability_index", "camera_state": 13},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "move_camera", "camera_state": 1},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "move_camera", "camera_state": 2},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "move_camera", "camera_state": 3},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "move_camera", "camera_state": 4},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "stop_move_camera", "camera_state": 111},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "stop_move_camera", "camera_state": 222},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "stop_move_camera", "camera_state": 333},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "stop_move_camera", "camera_state": 444},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "left_click", "camera_state": 1},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "left_click", "camera_state": 2},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "right_click", "camera_state": 1},
    {"type": "Control", "input_value": 1.0, "alt": False, "ctrl": False, "action": "right_click", "camera_state": 2},
]
ACTION_SPACE_SIZE = len(ACTION_SPACE)

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
    return torch.tensor(state, dtype=torch.float32)

def write_action_to_shared_memory(shared_memory, action_dict):
    """Writes an action dictionary (as JSON) into shared memory."""
    action_json_str = json.dumps(action_dict)
    action_bytes = action_json_str.encode("utf-16")[:512]  # Convert to UTF-16
    action_bytes += b"\x00" * (512 - len(action_bytes))  # Pad to 256 wchar_t
    shared_data = struct.pack(SHARED_MEMORY_FORMAT, False, True, b"\x00" * 512, action_bytes)
    shared_memory.seek(0)
    shared_memory.write(shared_data)
    shared_memory.flush()

def read_shared_memory(shared_memory):
    """Reads game state from shared memory."""
    shared_memory.seek(0)
    data = shared_memory.read(struct.calcsize(SHARED_MEMORY_FORMAT))
    bNewGameStateAvailable, bNewActionAvailable, game_state_bytes, action_bytes = struct.unpack(SHARED_MEMORY_FORMAT, data)
    game_state = game_state_bytes.decode("utf-16", errors="ignore").strip("\x00")
    return bNewGameStateAvailable, game_state

def play_action(state, q_network):
    """Selects the best action according to the trained Q-network."""
    with torch.no_grad():
        q_values = q_network(state.unsqueeze(0))
        action_index = q_values.max(1)[1].item()
    return ACTION_SPACE[action_index]

if __name__ == "__main__":
    print("[Play Agent] Connecting to shared memory...")

    try:
        shared_memory = mmap.mmap(-1, MEMORY_SIZE, tagname=SHARED_MEMORY_NAME, access=mmap.ACCESS_WRITE)
    except mmap.error as e:
        print(f"Error creating or accessing shared memory: {e}")
        print("Make sure the Unreal Engine application has created the shared memory with the same name.")
        raise

    # Load the trained network
    MODEL_PATH = "trained_network.pth" # Make sure this path is correct
    q_network = SimpleQNetwork(STATE_SPACE_SIZE, len(ACTION_SPACE))
    q_network.load_state_dict(torch.load(MODEL_PATH))
    q_network.eval()
    print(f"[Play Agent] Loaded trained network from {MODEL_PATH}")

    print("[Play Agent] Ready. Waiting for game state to play...")

    while True:
        time.sleep(0.1)
        bNewGameStateAvailable, game_state_str = read_shared_memory(shared_memory)

        if bNewGameStateAvailable and game_state_str:
            game_state_str = game_state_str.strip().replace('\x00', '')
            try:
                game_state_dict = json.loads(game_state_str)
                current_state_tensor = state_to_tensor(game_state_dict)

                # Select action using the loaded network (greedy policy)
                action = play_action(current_state_tensor, q_network)
                write_action_to_shared_memory(shared_memory, action)
                print(f"[Play Agent] Sent action: {action}")

            except json.JSONDecodeError:
                print("[Play Agent] Error decoding JSON.")
            except Exception as e:
                print(f"[Play Agent] An error occurred: {e}")
        else:
            time.sleep(0.01)