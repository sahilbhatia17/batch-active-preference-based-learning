from sampling import Sampler
import algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo
import sys
import demos

class Node():

    def __init__(self, content, reward_value=None, parents=None):
        self.content = content # the actual solution (e.g. a trace)
        self.reward_value = reward_value # the reward value, if applicable
        if parents is None:
            self.parents = []
        else:
            self.parents = parents # the parents, if applicable

    def add_parent(self, parent_node):
        self.parents.append(parent_node)


def get_node_reward(node):
    return node.reward_value

def sort_on_rewards(node_list):
    # if we don't have an ordering originally on the nodes, but we have reward values, sort on those.
    # assign parents based on the results of the sorting
    sorted_reward_list = sorted(node_list, key=get_node_reward, reverse=True)
    # list is now sorted by rewards from high to low, assign parents accordingly
    for nodeidx in range(1, len(node_list)):
        if node_list[nodeidx].reward_value is None:
            print("ERROR: FOUND NODE WITH NO REWARD VALUE")
        sorted_reward_list[nodeidx].add_parent(node_list[nodeidx - 1])
    return sorted_reward_list

