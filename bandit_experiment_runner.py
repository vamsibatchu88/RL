from typing import List

from core.bandit_alg import BanditAlg
from core.bandit_env import BanditEnvironment


class BanditExperimentRunner:
    """
    Runs a an experiment for a given bandit algorithm on a given bandit problem.
    """
    env: BanditEnvironment  # The multi-armed bandit environment
    alg: BanditAlg  # The algorithm to run
    actions: List[str]  # The actions selected on each step
    rewards: List[float]  # The rewards received on each step

    def __init__(self, env: BanditEnvironment, alg: BanditAlg):
        """
        Initializes the experiment runner.

        :param env: The multi-armed bandit problem.
        :param alg: The algorithm to run.
        """
        self.env = env
        self.alg = alg
        self.actions = []
        self.rewards = []

        self.alg.initialize(arms=self.env.get_arms())

    def run(self, num_steps: int):
        """
        Runs the experiment for the given number of steps.

        :param num_steps: The number of steps to run the algorithm for.
        """
        for i in range(num_steps):
            # Get selected action and reward, then update alg
            action = self.alg.get_next_action()
            reward = self.env.get_reward(action)
            self.alg.update_with_last_reward(reward)

            self.actions.append(action)
            self.rewards.append(reward)

    def reset(self):
        """
        Resets the experiment runner, as well as the algorithm.
        """
        self.actions = []
        self.rewards = []
        self.alg.reset()
