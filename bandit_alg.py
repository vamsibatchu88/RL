from abc import ABC, abstractmethod
from typing import List


class BanditAlg(ABC):
    """
    Defines the template for a multi-armed bandit algorithm.
    """
    arms: List[str]  # The list of arm names

    @abstractmethod
    def initialize(self, arms: List[str]):
        """
        Initializes the algorithm with the given list of arms.

        :param arms: The list of arms
        """
        pass

    @abstractmethod
    def get_next_action(self) -> str:
        """
        Gets the next arm to select.

        :return: The next action to perform.
        """
        pass

    @abstractmethod
    def update_with_last_reward(self, reward: float):
        """
        Updates the algorithm given the last reward received. Assumes that the
        last action selected has been stored by the algorithm.

        :param reward: The last reward received.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the bandit algorithm for a new experiment.
        """
        pass
