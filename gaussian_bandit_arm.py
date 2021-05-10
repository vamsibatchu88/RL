import random


class GaussianBanditArm:
    """
    A bandit arm whose reward is distributed like a Gaussian.
    """
    mean: float  # The mean of the reward for this arm.
    stdev: float  # The standard deviation for the reward for this arm.

    def __init__(self, name: str, mean: float, stdev: float):
        """
        Initializes the given bandit arm with the given mean
        and standard deviation.

        :param mean: The mean reward of the arm.
        :param stdev: The standard deviation for the reward of this arm.
        """
        self.name = name
        self.mean = mean
        self.stdev = stdev

    def get_reward(self) -> float:
        """
        Returns a reward for selecting this arm.

        :return: A reward from this arm.
        """
        return random.gauss(self.mean, self.stdev)
