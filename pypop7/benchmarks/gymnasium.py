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


class Ant(object):  # linear neural network
    """Control of Ant via a linear neural network.
    """
    def __init__(self):
        self.env = gym.make('Ant-v2')
        self.observation, _ = self.env.reset()
        self.action_dim = np.prod(self.env.action_space.shape)  # for action probability space

    def __call__(self, x):
        """Control of Ant via a linear neural network.

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
            self.observation, reward, terminated, truncated, _ = self.env.step(action)
            fitness -= reward
            if terminated or truncated:
                return fitness  # for minimization (rather than maximization)
        return fitness  # for minimization (rather than maximization)


class HalfCheetah(object):  # linear neural network
    """Control of HalfCheetah via a linear neural network.
    """
    def __init__(self):
        self.env = gym.make('HalfCheetah-v2')
        self.observation, _ = self.env.reset()
        self.action_dim = np.prod(self.env.action_space.shape)  # for action probability space

    def __call__(self, x):
        """Control of HalfCheetah via a linear neural network.

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
            self.observation, reward, terminated, truncated, _ = self.env.step(action)
            fitness -= reward
            if terminated or truncated:
                return fitness  # for minimization (rather than maximization)
        return fitness  # for minimization (rather than maximization)


class Hopper(object):  # linear neural network
    """Control of Hopper via a linear neural network.
    """
    def __init__(self):
        self.env = gym.make('Hopper-v2')
        self.observation, _ = self.env.reset()
        self.action_dim = np.prod(self.env.action_space.shape)  # for action probability space

    def __call__(self, x):
        """Control of Hopper via a linear neural network.

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
            self.observation, reward, terminated, truncated, _ = self.env.step(action)
            fitness -= reward
            if terminated or truncated:
                return fitness  # for minimization (rather than maximization)
        return fitness  # for minimization (rather than maximization)


class Humanoid(object):  # linear neural network
    """Control of Humanoid via a linear neural network.
    """
    def __init__(self):
        self.env = gym.make('Humanoid-v2')
        self.observation, _ = self.env.reset()
        self.action_dim = np.prod(self.env.action_space.shape)  # for action probability space

    def __call__(self, x):
        """Control of Humanoid via a linear neural network.

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
            self.observation, reward, terminated, truncated, _ = self.env.step(action)
            fitness -= reward
            if terminated or truncated:
                return fitness  # for minimization (rather than maximization)
        return fitness  # for minimization (rather than maximization)


class Swimmer(object):  # linear neural network
    """Control of Swimmer via a linear neural network.
    """
    def __init__(self):
        self.env = gym.make('Swimmer-v2')
        self.observation, _ = self.env.reset()
        self.action_dim = np.prod(self.env.action_space.shape)  # for action probability space

    def __call__(self, x):
        """Control of Swimmer via a linear neural network.

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
            self.observation, reward, terminated, truncated, _ = self.env.step(action)
            fitness -= reward
            if terminated or truncated:
                return fitness  # for minimization (rather than maximization)
        return fitness  # for minimization (rather than maximization)


class Walker2d(object):  # linear neural network
    """Control of Walker2d via a linear neural network.
    """
    def __init__(self):
        self.env = gym.make('Walker2d-v2')
        self.observation, _ = self.env.reset()
        self.action_dim = np.prod(self.env.action_space.shape)  # for action probability space

    def __call__(self, x):
        """Control of Walker2d via a linear neural network.

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
            self.observation, reward, terminated, truncated, _ = self.env.step(action)
            fitness -= reward
            if terminated or truncated:
                return fitness  # for minimization (rather than maximization)
        return fitness  # for minimization (rather than maximization)
