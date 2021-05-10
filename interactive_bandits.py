import random

from core.bandit_env import BanditEnvironment
from core.gaussian_bandit_env import GaussianBanditEnvironment


def get_random_gaussian_bandit_environment(
    num_arms: int,
    min_mean: float,
    max_mean: float,
    stdev: float,
) -> GaussianBanditEnvironment:
    """
    Gets a random Gaussian bandit environment with the given number of arms, range of expected
    rewards, and standard deviation.

    :param num_arms: The number of arms in the environment
    :param min_mean: The minimum expected reward
    :param max_mean: The maximum expected reward
    :param stdev: The standard deviation of all the arms.
    :return: A Gaussian bandit environment
    """
    assert num_arms >= 1
    assert min_mean <= max_mean
    assert stdev >= 0.

    env = GaussianBanditEnvironment()

    for i in range(num_arms):
        mean = random.uniform(min_mean, max_mean)
        env.add_arm(name=str(i + 1), mean=mean, stdev=stdev)

    return env


def run_interactive_bandits(env: BanditEnvironment):
    """
    Runs an interactive session with the bandit environment.

    :param env: The environment to interact with
    """
    arms = env.get_arms()
    total_reward = 0
    num_rounds = 0
    while True:
        prompt = "[ROUND " + str(num_rounds + 1) + "] "
        prompt += "Enter an arm from the list " + str(arms) + " (or 'exit'): "
        arm = input(prompt)
        if arm == 'exit':
            print("\tEXITING")
            print("\n\nTotal reward of", round(total_reward, 2), "in", num_rounds, "rounds")
            break
        elif arm in arms:
            reward = round(env.get_reward(arm), 2)
            print("\tReward received:", reward)
            total_reward += reward
            num_rounds += 1
        else:
            print("Invalid input")
