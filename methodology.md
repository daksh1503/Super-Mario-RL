Super Mario Bros Reinforcement Learning using Dueling Double DQN
==============================================================

1. Introduction
---------------
This implementation uses a sophisticated Deep Reinforcement Learning approach to train an agent to play Super Mario Bros. The methodology combines several state-of-the-art techniques to enhance learning efficiency and stability.

2. Literature Survey
-------------------
2.1 Deep Q-Networks (DQN)
- Original DQN paper (Mnih et al., 2015) introduced the combination of Q-learning with deep neural networks
- Key innovation: Experience replay and target networks for stable learning

2.2 Double DQN (DDQN)
- Introduced by van Hasselt et al. (2016)
- Addresses overestimation bias in traditional DQN
- Uses two networks for action selection and evaluation

2.3 Dueling Networks
- Wang et al. (2016) proposed separating value and advantage streams
- Improves learning efficiency by decomposing Q-values
- Particularly effective in environments with many actions

2.4 Prioritized Experience Replay (PER)
- Schaul et al. (2015) introduced prioritized sampling
- Samples important transitions more frequently
- Uses TD-error as a proxy for priority

3. Methodology
-------------
3.1 Network Architecture
- Input: 4 stacked frames (84x84 pixels)
- Convolutional layers: 3 layers with ReLU activation
- Dueling streams:
  * Value stream: Estimates state value
  * Advantage stream: Estimates action advantages

3.2 Training Process
- Environment: SuperMarioBros-v0
- Input preprocessing:
  * Grayscale conversion
  * Resolution downsampling
  * Frame stacking
- Hyperparameters:
  * Batch size: 512
  * Learning rate: 0.0005
  * Gamma (discount factor): 0.99
  * Memory size: 100,000 transitions
  * Target network update: Every 50 steps

3.3 Key Improvements
a) Double Q-learning
   - Reduces overestimation bias
   - Uses online network for action selection
   - Uses target network for value estimation

b) Prioritized Experience Replay
   - Prioritizes transitions based on TD-error
   - Uses importance sampling to correct bias
   - Progressive beta annealing

c) Epsilon Schedule
   - Starting epsilon: 1.0
   - Ending epsilon: 0.01
   - Decay rate: 0.99995

4. Evaluation
-------------
4.1 Metrics
- Average episode reward
- Stage completion rate
- Training stability
- Learning efficiency

4.2 Checkpoints
- Saved every 1000 episodes
- Includes:
  * Model states
  * Optimizer states
  * Training metrics
  * Current epsilon value

5. Implementation Details
------------------------
5.1 Environment Wrapper
- Custom wrapper for Mario environment
- Implements frame stacking
- Reward shaping:
  * Positive reward for forward progress
  * Penalty for death
  * Bonus for level completion

5.2 Training Infrastructure
- Supports CPU, CUDA, and MPS devices
- Automatic device selection
- Periodic model saving and evaluation

6. References
------------
1. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
2. Van Hasselt, H., et al. (2016). Deep reinforcement learning with double q-learning. AAAI, 30(1).
3. Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. ICML.
4. Schaul, T., et al. (2015). Prioritized experience replay. arXiv:1511.05952.

Note: This implementation combines these techniques to create a robust learning system for the Super Mario Bros environment, with careful consideration given to the interaction between different components and their impact on learning stability and efficiency. 