import mmap
import struct
import time
import json
import win32event
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Define shared memory name (same as Unreal)
SHARED_MEMORY_NAME = "Global\\UnrealRLSharedMemory"
MEMORY_SIZE = 4098    # Should match Unreal's size

# Structure format (must match Unreal's `SharedData`)
SHARED_MEMORY_FORMAT = "??" + "2048s" + "2048s"


# --- RL Agent Parameters ---
LEARNING_RATE = 0.01
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 1000
NUM_EPISODES = 10000  # <--- This line should be present
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 100
# --- RL Agent Parameters ---
# ... (previous parameters) ...

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

# ... (SimpleQNetwork and other functions remain similar, but action selection needs to change) ...
def state_to_tensor(game_state_dict):
    """Converts the game state dictionary to a PyTorch tensor."""
    state = [
        game_state_dict.get("MyUnits", 0),
        game_state_dict.get("EnemyUnits", 0),
        game_state_dict.get("MyHealth", 0.0),
        game_state_dict.get("EnemyHealth", 0.0),
        game_state_dict.get("MyAttack", 0.0),
        game_state_dict.get("EnemyAttack", 0.0),
        # You could potentially include some representation of the position data here
        # For example, the relative distance or a discretized grid.
    ]
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Unsqueeze to add batch dimension


def read_shared_memory(shared_memory):
    """Reads game state from shared memory."""
    shared_memory.seek(0)
    data = shared_memory.read(struct.calcsize(SHARED_MEMORY_FORMAT))
    bNewGameStateAvailable, bNewActionAvailable, game_state_bytes, action_bytes = struct.unpack(SHARED_MEMORY_FORMAT, data)

    # Decode the received game state
    game_state = game_state_bytes.decode("utf-16", errors="ignore").strip("\x00")
    return bNewGameStateAvailable, game_state


def select_action(state, episode):
    """Selects an action from the defined action space."""
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
              torch.exp(torch.tensor(-1. * episode / EPSILON_DECAY))
    if random.random() > epsilon.item(): # Use .item() to get the float value for comparison
        # For now, just select a random action index
        return random.randrange(ACTION_SPACE_SIZE)
    else:
        # In a real scenario, you would use the Q-network to select the best action
        # based on the state. For this basic example, we'll still use random.
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
    while episode_number < NUM_EPISODES:
        time.sleep(0.2)  # Match Unreal's update rate

        # Read game state
        bNewGameStateAvailable, game_state_str = read_shared_memory(shared_memory)

        if bNewGameStateAvailable and game_state_str:
            print(f"[RL Agent] Received Game State: {game_state_str}")
            game_state_str = game_state_str.strip() # Added this line
            print(f"[RL Agent] Raw Game State: {repr(game_state_str)}") # Added this line
            try:
                game_state_dict = json.loads(game_state_str)
                current_state_tensor = state_to_tensor(game_state_dict)

                # Select action using the RL agent
                action_index = select_action(current_state_tensor, episode_number)
                chosen_action = ACTION_SPACE[action_index]
                print(f"[RL Agent] Decided Action (Index {action_index}): {chosen_action}")

                # For actions that require a target unit (e.g., LeftClick), you'll need logic
                # to determine which unit to target based on the game state.
                # This is a placeholder for now.
                if chosen_action["type"] in ["LeftClick", "RightClick"]:
                    # Example: Target the first enemy unit if available
                    if game_state_dict.get("EnemyUnits", 0) > 0:
                        chosen_action["TargetUnitId"] = -1 # Replace with actual logic to get a unit ID
                    else:
                        continue # Skip action if no target

                # Write the action back to shared memory as JSON
                write_action_to_shared_memory(shared_memory, chosen_action)

                episode_number += 1

            except json.JSONDecodeError:
                print("[RL Agent] Error decoding JSON.")
            except Exception as e:
                print(f"[RL Agent] An error occurred: {e}")

    print("[RL Agent] Training finished (for this basic example, it just ran for a number of episodes).")