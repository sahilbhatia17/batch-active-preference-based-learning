from sampling import Sampler
import algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo, get_membership_feedback_auto, get_feedback_auto
import sys
import demos
import math
import lattice
from sklearn import svm
import z3gi

KEEPING_SPEED_BINS = 10.0
COLLISION_BINS = 10.0
DRIVING_FEATURES = (1, 3)
TASK_MAP = {"driver" : DRIVING_FEATURES}
def trajectory_to_discrete_alphabet(trajectory, task="driver", upperbound=1.0):
    # input is a feature of trajectories
    token_sequence = []

    relevant_features = trajectory[:, TASK_MAP[task]]
    for feature_pair in relevant_features:
        # bin these features
        tuple_symbol = []
        for feature_item in feature_pair: # the alphabet is all combinations of bins for each feature.
            tuple_symbol.append(min(convert_to_dec(feature_item) + .1, upperbound))  # cap it at 1 as the upper bound
        token_sequence.append(str(tuple(tuple_symbol)))
    return token_sequence

def get_dfa_utils(task="driver"):
    # add support later on for other tasks if needed
    alphabet = []
    collision_increment = 1.0 / COLLISION_BINS
    speed_increment = 1.0 / KEEPING_SPEED_BINS
    collision_val = 0.0
    while collision_val < 1:
        collision_val += 0.1
        speed_val = 0.0
        while speed_val < 1:
            speed_val += 0.1
            alphabet.append(str((convert_to_dec(speed_val), convert_to_dec(collision_val))))
    return alphabet

def convert_to_dec(num):
    return math.trunc(num * 10) / 10

def create_dfa_dataset_file(true_reward_weights, true_reward_boundary, sample_trajectories, task="driver", filename="dfa_file"):
    # first, open a file, and write the heading to it
    # the first line should be the entire alphabet as well as whether or not the trajectory is accepted
    # for this purpose, we assume that this first trajectory is not accepted.
    alphabet_set = get_dfa_utils(task)
    all_lines = []
    firstline = ['0'] + alphabet_set
    all_lines.append(' '.join(firstline))
    # now, get the result of the trajectories (which should be nodes) and add their corresponding lines to the list
    for trajectory_node in sample_trajectories:
        if np.sum(trajectory_node.features * true_reward_weights) > true_reward_boundary:
            label = ['1']
        else:
            label = ['0']
        nextline = label + trajectory_to_discrete_alphabet(trajectory_node.content, task)
        all_lines.append(' '.join(nextline))
    print("writing file.")
    fle = open(filename, "w")
    fle.writelines(all_lines)
    fle.close()

def collect_trajectories_wrapper(samplemethod, num_samples, reward_values, task="driver", membership_bound=-1):
    simulation_object = create_env(task)
    if membership_bound == -1:
        return demos.collect_trajectories(simulation_object, samplemethod, num_samples, reward_values)
    else:
        return demos.collect_member_trajectories(simulation_object, samplemethod, num_samples, reward_values, membership_bound)
