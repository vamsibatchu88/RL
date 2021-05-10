from typing import List, Dict
import random

from core.bandit_alg import BanditAlg


class EpsilonGreedyAlg(BanditAlg):
    """
    The epsilon-greedy algorithm.
    """
    arms: List[str]  # The list of arm names
    arm_indices: Dict[str, float]  # The indices of each arm in the q-values and action counts
    epsilon: float  # The value of epsilon
    q_values: List[float]  # A list of the q-values for each action
    action_counts: List[int]  # A list of the action counts
    last_action: str  # The last action selected

    def __init__(self, epsilon: float):
        """
        Creates an instance of the algorithm.

        :param epsilon: The value of epsilon
        """
        assert 0 <= epsilon <= 1

        self.epsilon = epsilon

        self.arms = None
        self.arm_indices = None
        self.q_values = None
        self.action_counts = None
        self.last_action = None

    def initialize(self, arms: List[str]):
        """
        Initializes the q-value and count storing tables.

        :param arms: The list of arms that can be selected.
        """
        assert len(arms) > 0

        self.last_action = None
        self.arms = arms
        self.q_values = [0.0] * len(self.arms)
        self.action_counts = [0] * len(self.arms)

        self.arm_indices = {}
        for i, arm in enumerate(self.arms):
            self.arm_indices[arm] = i

    def get_greedy_action_index(self) -> int:
        """
        Gets the index of a greedy action.

        :return: The index of a greedy action.
        """
        best_arms = [0]
        best_q_value = self.q_values[0]

        for i in range(1, len(self.arms)):
            q = self.q_values[i]

            if q > best_q_value:
                best_arms = [i]
                best_q_value = q
            elif q == best_q_value:
                best_arms.append(i)

        index = random.randint(0, len(best_arms) - 1)
        return best_arms[index]

    def get_next_action(self) -> str:
        """
        Gets the next action selected by the algorithm.

        :return: The name of the arm to select.
        """
        if random.uniform(0.0, 1.0) <= self.epsilon:
            arm_index = random.randint(0, len(self.arms) - 1)
        else:
            arm_index = self.get_greedy_action_index()

        self.last_action = self.arms[arm_index]
        return self.last_action

    def update_with_last_reward(self, reward: float):
        """
        Updates the algorithm given the last reward received. Assumes that the
        last action selected has been stored by the algorithm.

        :param reward: The last reward received.
        """
        assert self.last_action in self.arm_indices
        arm_index = self.arm_indices[self.last_action]

        self.action_counts[arm_index] += 1
        old_q = self.q_values[arm_index]
        step_size = 1 / self.action_counts[arm_index]

        self.q_values[arm_index] = old_q + step_size * (reward - old_q)

    def reset(self):
        """
        Resets the bandit algorithm for a new experiment.
        """
        self.initialize(self.arms)

    def change_epsilon(self, epsilon: float):
        """
        Changes the value of epsilon and resets the algorithm.

        :param epsilon: The new value of epsilon.
        """
        assert 0 <= epsilon <= 1.
        self.epsilon = epsilon
