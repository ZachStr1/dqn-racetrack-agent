🏎️ Deep Q-Learning Race Track Agent

A reinforcement learning project where a Deep Q-Network (DQN) agent learns to drive autonomously around a custom 2D race track using ray-based perception and physics-based vehicle dynamics.

This project was built to explore applied deep reinforcement learning, reward shaping, and environment design from scratch.

⸻

🎥 Demo

(Add a short GIF or screenshot here later)

The agent:
	•	Uses ray sensors to detect walls
	•	Learns throttle + steering control
	•	Receives reward for forward progress
	•	Completes full laps without checkpoints
	•	Improves lap time through training

⸻

🧠 Project Overview

This project implements:
	•	Custom 2D race environment (no gym dependency)
	•	Physics-based car model
	•	Ray-cast perception system
	•	Deep Q-Network (PyTorch)
	•	Experience replay buffer
	•	Epsilon-greedy exploration
	•	Reward shaping with lap detection
	•	Training metrics logging

Unlike many RL tutorials, this environment was built entirely from scratch — including:
	•	Collision detection
	•	Waypoint-based progress tracking
	•	Start/finish line lap detection
	•	Stuck detection and no-progress termination

⸻

🏗️ Environment Design

Observation Space

Each state consists of:
	•	13 forward-facing ray distances (normalized 0–1)
	•	1 normalized speed value

Total state dimension: 14

Action Space (Discrete: 9 actions)

Action	Description
0	Steer Left
1	Steer Right
2	Throttle Forward
3	Throttle Reverse
4	Forward + Left
5	Forward + Right
6	Reverse + Left
7	Reverse + Right
8	No Input


⸻

🎯 Reward Function

Reward is composed of:
	•	✅ Forward progress along track waypoints
	•	➕ Small forward velocity incentive
	•	➖ Time penalty per step
	•	➖ Wall proximity penalty
	•	➖ Crash penalty
	•	🎉 Lap completion bonus

Progress is calculated using closest waypoint indexing and normalized over total track length.

Lap detection uses start/finish line intersection — no artificial checkpoints.

⸻

🧪 Training Setup
	•	Algorithm: Deep Q-Network (DQN)
	•	Framework: PyTorch
	•	Device: Apple MPS (Metal GPU acceleration)
	•	Replay Buffer Size: 50,000
	•	Max Steps: 50,000 per training run
	•	Epsilon Decay: Linear
	•	Physics timestep: Fixed 1/60s

Training logs example:

step=50000 eps=0.050 buffer=50000 device=mps laps=5 crashes=187

After training, the agent consistently completes laps autonomously.

Best lap time achieved:

8.35 seconds


⸻

🖥️ How To Run

1️⃣ Create virtual environment

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2️⃣ Train agent

python -m src.train_dqn

3️⃣ View trained policy

MODEL_PATH="runs/<timestamp>/dqn_final.pt" python -m src.main_view_policy


⸻

📂 Project Structure

src/
│
├── env/
│   ├── racetrack_env.py
│   ├── track.py
│   ├── car.py
│   └── utils.py
│
├── rl/
│   ├── dqn_agent.py
│   └── replay_buffer.py
│
├── train_dqn.py
└── main_view_policy.py


⸻

🧩 Key Engineering Challenges
	•	Stabilizing DQN training
	•	Preventing reward exploitation
	•	Designing smooth progress measurement
	•	Eliminating checkpoint hacks
	•	Avoiding spinning / wall-hugging behavior
	•	Ensuring stable lap detection

⸻

🚀 Future Improvements
	•	Double DQN
	•	Prioritized replay
	•	Continuous control (DDPG / PPO)
	•	Curved or procedurally generated tracks
	•	Curriculum learning
	•	Model-based RL experiments

⸻

📚 What I Learned
	•	How reward shaping dramatically affects agent behavior
	•	Why checkpoint systems can produce shortcut exploitation
	•	The importance of environment design in RL
	•	Debugging unstable Q-value explosions
	•	Practical reinforcement learning beyond textbook examples

⸻
