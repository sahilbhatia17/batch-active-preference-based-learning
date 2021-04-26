from sampling import Sampler
import algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo, get_membership_feedback
import sys
import math
import lattice
from sklearn import svm
from lstar import learn_dfa

KEEPING_SPEED_BINS = 10
COLLISION_BINS = 10
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
            tuple_symbol.add(min(round(feature_item, 1) + .1, upperbound)) # cap it at 1 as the upper bound
        token_sequence.append(str(tuple(tuple_symbol)))
    return token_sequence

def get_dfa_utils(task="driver"):
    # add support later on for other tasks if needed
    alphabet = []
    collision_increment = 1.0 / COLLISION_BINS
    speed_increment = 1.0 / KEEPING_SPEED_BINS
    collision_val = 0.0
    while collision_val <= 1.0:
        collision_val += 0.1
        speed_val = 0.0
        while speed_val <= 1.0:
            speed_val += 0.1
            alphabet.append(str((speed_val, collision_val)))
    return set(alphabet)

