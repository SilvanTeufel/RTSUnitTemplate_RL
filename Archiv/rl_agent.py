import mmap
import struct
import time
import json
import win32event
# import win32file

# Define shared memory name (same as Unreal)
SHARED_MEMORY_NAME = "Global\\UnrealRLSharedMemory"
MEMORY_SIZE = 1026  # Should match Unreal's size

# Structure format (must match Unreal's `SharedData`)
# 1 boolean (1 byte), 1 boolean (1 byte), GameState (256 wchar_t), Action (256 wchar_t)
SHARED_MEMORY_FORMAT = "??" + "512s" + "512s"

def read_shared_memory(shared_memory):
    """Reads game state from shared memory."""
    shared_memory.seek(0)
    data = shared_memory.read(struct.calcsize(SHARED_MEMORY_FORMAT))
    bNewGameStateAvailable, bNewActionAvailable, game_state_bytes, action_bytes = struct.unpack(SHARED_MEMORY_FORMAT, data)

    # Decode the received game state
    game_state = game_state_bytes.decode("utf-16", errors="ignore").strip("\x00")
    
    return bNewGameStateAvailable, game_state

def write_shared_memory(shared_memory, action_str):
    """Writes an action string into shared memory."""
    action_bytes = action_str.encode("utf-16")[:512]  # Convert to UTF-16
    action_bytes += b"\x00" * (512 - len(action_bytes))  # Pad to 256 wchar_t

    # Create a new shared data block with action
    shared_data = struct.pack(SHARED_MEMORY_FORMAT, False, True, b"\x00" * 512, action_bytes)

    shared_memory.seek(0)
    shared_memory.write(shared_data)
    shared_memory.flush()

def dummy_rl_decision(game_state):
    """A simple decision-making function. Replace with an actual RL model."""
    try:
        game_data = json.loads(game_state)
        if game_data["MyUnits"] > game_data["EnemyUnits"]:
            return "Attack"
        else:
            return "Defend"
    except:
        return "Idle"

if __name__ == "__main__":
    print("[RL Agent] Connecting to shared memory...")

    try:
        shared_memory = mmap.mmap(-1, MEMORY_SIZE, tagname=SHARED_MEMORY_NAME, access=mmap.ACCESS_WRITE)
    except mmap.error as e:
        print(f"Error creating or accessing shared memory: {e}")
        print("Make sure the Unreal Engine application has created the shared memory with the same name.")
        raise

    print("[RL Agent] Ready. Waiting for game state...")

    while True:
        time.sleep(0.2)  # Match Unreal's update rate

        # Read game state
        bNewGameStateAvailable, game_state = read_shared_memory(shared_memory)

        if bNewGameStateAvailable and game_state:
            print(f"[RL Agent] Received Game State: {game_state}")

            # Generate an action based on the game state
            action = dummy_rl_decision(game_state)
            print(f"[RL Agent] Decided Action: {action}")

            # Write the action back to shared memory
            write_shared_memory(shared_memory, action)
