from abc import ABC, abstractmethod


class BanditEnvironment(ABC):
    """
    Defines a template for multi-armed bandit environment with arms whose rewards are distributed
    like a Gaussian.
    """
    @abstractmethod
    def get_arms(self) -> str:
        """
        Gets the list of names of arms that can be selected

        :return: The list of arm names.
        """
        pass

    @abstractmethod
    def get_reward(self, action_name: str) -> float:
        """
        Gets the reward given the selected action name.

        :param action_name: The action selected.
        :return: The reward received for that selected action.
        """
        pass
