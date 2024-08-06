# Mixture of Experts (MoE) for Reinforcement Learning in Atari Environments

This repository implements a Mixture of Experts (MoE) framework for reinforcement learning in an Atari environment, combining Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) strategies with a vision processing network and a gating network to optimize action selection.

## Dependencies

    Python 3.x
    PyTorch
    OpenAI Gym
    NumPy
    OpenCV
    Matplotlib

## Components
1. Environment Preprocessing

Class PreprocessAtari:

    Purpose: Preprocesses Atari game frames to make them suitable for neural network input.
    Methods:
        preprocess: Converts frames to grayscale and resizes them to 84x84.
        reset and step: Apply preprocessing to the environment's reset and step functions.

2. Neural Network Models

Class AdaptiveVisionExpert:

    Purpose: Processes raw image inputs using convolutional layers.
    Architecture: Three convolutional layers with batch normalization and ReLU activations.

Class DQNExpert:

    Purpose: Provides a fully connected neural network for the DQN strategy.
    Architecture: Three fully connected layers with ReLU activations.

Class PPOExpert:

    Purpose: Implements separate actor and critic networks for the PPO strategy.
    Architecture: Actor and critic each have three fully connected layers with ReLU activations.

Class GatingNetwork:

    Purpose: Determines the probabilities of using different experts (DQN or PPO).
    Architecture: Two fully connected layers with ReLU activation and softmax output.

3. Mixture of Experts (MoE) Framework

Class MoE:

    Purpose: Integrates the vision expert, DQN expert, PPO expert, and gating network.
    Key Methods:
        select_action: Chooses an action based on the gating network's probabilities.
        update: Trains the vision expert, DQN expert, PPO expert, and gating network using experiences from the replay buffer.
        evaluate: Tests the agent over multiple episodes and returns mean and standard deviation of rewards.
        train: Main training loop that runs for a specified number of episodes, updating the model and periodically evaluating its performance.
        save_model and load_model: Save and load model parameters.
