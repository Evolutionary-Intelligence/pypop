import numpy as np
# https://github.com/Farama-Foundation/Gymnasium
import gymnasium as gym


class Cartpole(object):  # linear neural network
    """Control of Cartpole via a linear neural network.
    """
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode='human')
        self.observation, _ = self.env.reset()
        self.action_dim = 2  # for action probability space

    def __call__(self, x):
        """Control of Cartpole via a linear neural network.

        Parameters
        ----------
        x : ndarray
            input vector.

        Returns
        -------
        fitness : float
                  negative reward (for minimization rather than maximization).
        """
        fitness = 0
        self.observation, _ = self.env.reset()
        for i in range(1000):
            action = np.matmul(x.reshape(self.action_dim, -1), self.observation[:, np.newaxis])
            actions = np.sum(action)
            # seen as a probability
            prob_left, prob_right = action[0] / actions, action[1] / actions
            action = 1 if prob_left < prob_right else 0
            self.observation, reward, terminated, truncated, _ = self.env.step(action)
            fitness -= reward
            if terminated or truncated:
                return fitness  # for minimization (rather than maximization)
        return fitness  # for minimization (rather than maximization)
