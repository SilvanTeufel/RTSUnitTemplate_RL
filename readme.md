Of course\! Based on the file names shown in your image, here is the corrected `README.md`.

-----

# Unreal Engine & PyTorch Reinforcement Learning Agent

This project implements a Reinforcement Learning (RL) agent in Python using PyTorch. The agent is designed to interact with and learn to play a game running in Unreal Engine. Communication between the Python agent and the Unreal Engine game is achieved through shared memory.

The repository contains several key files:

1.  **`rl_agent_deepq_5.py`**: A script for training a Deep Q-Network (DQN) agent from scratch.
2.  **`play_game.py`**: A script for running inference with a pre-trained agent model.
3.  **`weight_averageing.py`**: A utility for model weight manipulation, likely for creating an averaged model from several checkpoints.

## Features

  - **Reinforcement Learning**: Uses a Deep Q-Network (DQN) with a replay buffer and a target network for stable training.
  - **PyTorch Integration**: The neural network model is built and trained using PyTorch.
  - **Shared Memory Communication**: Efficient, high-speed communication with an Unreal Engine application using `mmap` on Windows.
  - **Dynamic Configuration**: Supports multiple instances via command-line `TeamId` arguments.
  - **Complex Action & State Spaces**: Designed to handle a variety of game states (unit counts, health, positions, resources) and a comprehensive set of possible actions (camera movement, unit selection, ability usage, etc.).
  - **Sophisticated Reward Shaping**: The reward function encourages complex behaviors like resource gathering, combat effectiveness, and strategic positioning.

## How It Works

The system operates in a continuous loop between the Unreal Engine game and the Python script:

1.  **Unreal Engine (Game)**:

      * Collects the current game state (e.g., unit positions, health, resource counts) and serializes it into a JSON string.
      * Writes this `GameState` JSON into a designated shared memory block.
      * Sets a flag (`bNewGameStateAvailable`) to `True`.
      * Waits for the Python agent to provide an action.

2.  **Python (RL Agent)**:

      * Continuously monitors the shared memory for the `bNewGameStateAvailable` flag.
      * When a new state is available, it reads and deserializes the `GameState` JSON.
      * Processes the state into a tensor suitable for the neural network.
      * **During Training (`rl_agent_deepq_5.py`)**: It calculates a reward based on the change from the previous state, stores the experience in a replay buffer, and trains the Q-network. It uses an epsilon-greedy policy to select the next action.
      * **During Inference (`play_game.py`)**: It feeds the state to the loaded, pre-trained network and greedily selects the action with the highest predicted Q-value.
      * Serializes the chosen action into a JSON string.
      * Writes the `Action` JSON back into the shared memory and sets the `bNewActionAvailable` flag to `True`.

3.  **Unreal Engine (Game)**:

      * Detects the `bNewActionAvailable` flag.
      * Reads and deserializes the `Action` JSON.
      * Executes the action within the game simulation.
      * The loop repeats.

## Prerequisites

  - Python 3.7+
  - Windows Operating System (due to `mmap` and `win32event` usage)
  - An Unreal Engine project configured to communicate via the specified shared memory structure.
  - Required Python libraries. You can install them using pip:

<!-- end list -->

```bash
pip install torch torchvision torchaudio
pip install pywin32
```

## Usage Instructions

### 🚨 Important: Run as Administrator

To create or access a **global** shared memory object (`Global\\...`), both the Unreal Engine Editor (or the packaged game) and the Python script must be run with **administrator privileges**.

  - **For Unreal Engine**: Right-click the Epic Games Launcher or your `UE4Editor.exe`/`UE5Editor.exe` and select "Run as administrator".
  - **For Python**: Open Command Prompt or PowerShell as an administrator to run the scripts.

-----

### 1\. Training the Agent

The `rl_agent_deepq_5.py` script trains the model and saves its progress.

1.  **Start Unreal Engine**: Launch your Unreal Engine project as an **administrator**. Start the game/simulation. The game should create the shared memory block and wait for a connection.

2.  **Run the Training Script**: Open an **administrator** command prompt, navigate to the project directory, and run the script with a unique `TeamId`. The `TeamId` is used to create a unique shared memory name, allowing multiple agent-game instances to run simultaneously without interfering with each other.

    ```bash
    python rl_agent_deepq_5.py <TeamId>
    ```

    **Example:**

    ```bash
    # For Team 1
    python rl_agent_deepq_5.py 1
    ```

3.  **Monitor Training**: The script will print its status, including the current episode, total steps, and rewards. It will periodically save the trained model weights to `trained_network.pth`.

### 2\. Playing with the Trained Agent

The `play_game.py` script loads the `trained_network.pth` file to play the game using the learned policy.

**Note**: The provided `play_game.py` script may use a hardcoded shared memory name (e.g., `Global\UnrealRLSharedMemory`). If your training script and Unreal project use a `TeamId` (e.g., `Global\UnrealRLSharedMemory_TeamId_1`), you **must** modify the `SHARED_MEMORY_NAME` variable in `play_game.py` to match it.

1.  **Ensure a Trained Model Exists**: Make sure you have a `trained_network.pth` file in the same directory.

2.  **Start Unreal Engine**: Launch your Unreal Engine project as an **administrator** and start the game/simulation.

3.  **Run the Play Script**: Open an **administrator** command prompt and run the script.

    ```bash
    python play_game.py
    ```

4.  **Observe**: The agent will now play the game by selecting the best action it knows for every game state it receives, printing the action it sends to the console.

## Code Breakdown

### File Descriptions

  - **`rl_agent_deepq_5.py`**: Contains the complete logic for a DQN agent, including the Q-network, target network, replay buffer, optimizer, and the main training loop. It handles both exploration (taking random actions) and exploitation (using the network's knowledge).
  - **`play_game.py`**: A lightweight script that loads a saved `.pth` model file and performs inference. It only uses the network for exploitation (always choosing the best action) and does not perform any training or exploration.
  - **`trained_network.pth`**: The default output file for the trained model weights, saved by `rl_agent_deepq_5.py` and loaded by `play_game.py`.
  - **`weight_averageing.py`**: A utility script, likely used for Stochastic Weight Averaging (SWA) or averaging weights from different training checkpoints to create a more robust final model.
  - **`readme.md`**: This documentation file.

### Key Concepts

#### State Space (`STATE_SPACE_SIZE = 21`)

The state is a vector of 21 numerical values representing the game world from the agent's perspective. This includes:

  - Unit counts (friendly and enemy)
  - Total unit health and attack power
  - The agent's 3D position
  - Average 3D positions of friendly and enemy units
  - Counts for 6 different types of resources

#### Action Space (`ACTION_SPACE`)

This is a predefined list of dictionaries, where each dictionary represents a possible action the agent can take. Actions include:

  - Selecting unit groups (`ctrl_select`, `alt_select`)
  - Using abilities (`switch_camera_state_ability`)
  - Managing resources (`resource_management`)
  - Moving the camera (`move_camera`)
  - Mouse clicks (`left_click`, `right_click`)

#### Reward Function (`extract_reward`)

The agent's learning is guided by the reward function. This function is carefully shaped to promote desirable behaviors:

  - **Positive Rewards**:
      - `+0.2` for decreasing the number of enemy units.
      - `+0.1` for increasing the number of own units.
      - `+0.05` for gathering any type of resource.
      - `+0.03` for spending resources (implies unit creation or upgrades).
      - Bonus rewards for dealing damage to enemies.
      - Proximity-based rewards to encourage engaging enemies and supporting allies.
  - **Negative Rewards**:
      - `-0.3` for losing one of its own units.
      - Small penalty for taking damage.
      - `-0.03` constant time penalty each step to encourage efficiency.

### Configuration Parameters (`rl_agent_deepq_5.py`)

You can modify the agent's learning behavior by tweaking these parameters at the top of the `rl_agent_deepq_5.py` script:

  - `LEARNING_RATE`: How much to adjust the network weights on each update.
  - `GAMMA`: Discount factor for future rewards. A value closer to 1 makes the agent more farsighted.
  - `EPSILON_START`/`_END`/`_DECAY`: Controls the exploration-exploitation tradeoff. The agent starts with a high probability of exploring (`EPSILON_START`) and gradually decreases it.
  - `NUM_EPISODES`: Total number of training episodes.
  - `BATCH_SIZE`: Number of experiences to sample from the replay buffer for each training step.
  - `TARGET_UPDATE_FREQUENCY`: How often to update the target network's weights.
  - `SAVE_FREQUENCY`: How many episodes to wait before saving a checkpoint of the model.