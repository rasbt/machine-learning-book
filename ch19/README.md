
##  Chapter 19: Reinforcement Learning for Decision Making in Complex Environments


### Chapter Outline

- Introduction: learning from experience
  - Understanding reinforcement learning
  - Defining the agent-environment interface of a reinforcement learning system
  - The theoretical foundations of RL
    - Markov decision processes
    - The mathematical formulation of Markov decision processes
    - Visualization of a Markov process
    - Episodic versus continuing tasks
  - RL terminology: return, policy, and value function
    - The return
    - Policy
    - Value function
  - Dynamic programming using the Bellman equation
- Reinforcement learning algorithms
  - Dynamic programming
    - Policy evaluation – predicting the value function with dynamic programmin
    - Improving the policy using the estimated value function
    - Policy iteration
    - Value iteration
  - Reinforcement learning with Monte Carlo
    - State-value function estimation using MC
    - Action-value function estimation using MC
    - Finding an optimal policy using MC control
    - Policy improvement – computing the greedy policy from the action-value function
  - Temporal difference learning
    - TD prediction
    - On-policy TD control (SARSA)
    - Off-policy TD control (Q-learning)
- Implementing our first RL algorithm
  - Introducing the OpenAI Gym toolkit
    - Working with the existing environments in OpenAI Gym
  - A grid world example
    - Implementing the grid world environment in OpenAI Gym
  - Solving the grid world problem with Q-learning
    - Implementing the Q-learning algorithm
- A glance at deep Q-learning
  - Training a DQN model according to the Q-learning algorithm
    - Replay memory
    - Determining the target values for computing the loss
  - Implementing a deep Q-learning algorithm
- Chapter and book summary

**Please refer to the [README.md](../ch01/README.md) file in [`../ch01`](../ch01) for more information about running the code examples.**


