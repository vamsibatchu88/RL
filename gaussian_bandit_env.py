from typing import Dict, List

from core.bandit_env import BanditEnvironment
from core.gaussian_bandit_arm import GaussianBanditArm


class GaussianBanditEnvironment(BanditEnvironment):
    """
    Defines a multi-armed bandit environment with arms whose rewards are distributed like a Gaussian.
    """
    arms: Dict[str, GaussianBanditArm]  # The arms in this bandit problem

    def __init__(self):
        """
        Initializes the bandit problem with an empty list of bandit arms.
        """
        self.arms = {}

    def add_arm(self, name: str, mean: float, stdev: float):
        """
        Adds an arm to the problem such that it has the given name, mean,
        and standard deviation.

        :param name: The name of the bandit.
        :param mean: The mean reward for selecting this bandit.
        :param stdev: The standard deviations of the rewards for this bandit.
        """
        assert stdev >= 0.
        assert name not in self.arms

        self.arms[name] = GaussianBanditArm(name=name, mean=mean, stdev=stdev)

    def get_arms(self) -> str:
        """
        Gets the list of names of arms that can be selected

        :return: The list of arm names.
        """
        return list(self.arms.keys())

    def get_reward(self, action_name: str) -> float:
        """
        Gets the reward given the selected action name.

        :param action_name: The action selected.
        :return: The reward received for that selected action.
        """
        assert action_name in self.arms

        return self.arms[action_name].get_reward()

    @classmethod
    def from_means_and_stdev(
            cls,
            arm_names: List[str],
            means: List[str],
            stdevs: float
    ) -> "GaussianBanditEnvironment":
        """
        Gets a Gaussian arm environment such that there is one arm for each mean
        in the means list with the corresponding

        :param arm_names: A list with the names of the arms
        :param means: A list with the means for the arms
        :param stdevs: A list with the standard deviation for the arms
        :return: A Gaussian bandit environment corresponding to the given input
        """

        assert len(arm_names) == len(means)
        assert len(arm_names) == len(stdevs)

        env = cls()
        for i in range(len(arm_names)):
            env.add_arm(name=arm_names[i], mean=means[i], stdev=stdevs[i])
        return env
