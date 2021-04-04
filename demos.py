from sampling import Sampler
import algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo, get_membership_feedback
import sys
import lattice

def batch(task, method, N, M, b):
    if N % b != 0:
        print('N must be divisible to b')
        exit(0)
    B = 20*b

    simulation_object = create_env(task)
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []
    inputA_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*simulation_object.feed_size))
    inputB_set = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(b, 2*simulation_object.feed_size))
    for j in range(b):
        input_A = inputA_set[j]
        input_B = inputB_set[j]
        psi, s = get_feedback(simulation_object, input_A, input_B)
        psi_set.append(psi)
        s_set.append(s)
    i = b
    while i < N:
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples,axis=0)
        print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        print('Samples so far: ' + str(i))
        inputA_set, inputB_set = run_algo(method, simulation_object, w_samples, b, B)
        for j in range(b):
            input_A = inputA_set[j]
            input_B = inputB_set[j]
            psi, s = get_feedback(simulation_object, input_B, input_A)
            psi_set.append(psi)
            s_set.append(s)
        i += b
    w_sampler.A = psi_set
    w_sampler.y = np.array(s_set).reshape(-1,1)
    w_samples = w_sampler.sample(M)
    mean_w_samples = np.mean(w_samples, axis=0)
    print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))



def nonbatch(task, method, N, M):
    simulation_object = create_env(task)
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []
    input_A = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(2*simulation_object.feed_size))
    input_B = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(2*simulation_object.feed_size))
    psi, s = get_feedback(simulation_object, input_A, input_B) # psi is the difference, s is the 1 or -1 signal
    psi_set.append(psi)
    s_set.append(s)
    for i in range(1, N):
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples,axis=0)
        print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        input_A, input_B = run_algo(method, simulation_object, w_samples)
        psi, s = get_feedback(simulation_object, input_A, input_B)
        psi_set.append(psi)
        s_set.append(s)
    w_sampler.A = psi_set
    w_sampler.y = np.array(s_set).reshape(-1,1)
    w_samples = w_sampler.sample(M)
    print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))

def membership_threshold(num_weighted_samples, num_random_samples, reward_values, task='driver', method="nonbatch"):
    # first, sample the trajectories from the distribution\
    simulation_object = create_env(task)
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    # set the reward weights of the sampler

    full_traj_set = []
    # sample trajectories from the weighted samples
    def add_traj(samplemethod):

        sample_A, sample_B = run_algo(samplemethod, simulation_object, reward_values.reshape(1,-1))
        simulation_object.feed(sample_A)
        phi_A = simulation_object.get_features()
        # now, compute the reward for each sample
        reward_A = np.sum(reward_values * phi_A)
        full_traj_set.append(lattice.Node(sample_A, reward_value=reward_A))
    # first, sample trajectories using the given weights
    for idx in range(num_weighted_samples):
        add_traj(method)
    # now, sample trajectories from the random samples
    for idx in range(num_random_samples):
        add_traj("random")
    # sort the trajectories by reward
    sorted_lattice = lattice.sort_on_rewards(full_traj_set)

    #now, begin getting membership query feedback on things
    remainder_to_search = sorted_lattice
    while len(remainder_to_search) > 1:
        current_idx = int(len(remainder_to_search) / 2)
        current_candidate = remainder_to_search[current_idx]
        # ask: is this a member of the preferred set?
        print("current reward boundary is: {}".format(current_candidate.reward_value))
        response = get_membership_feedback(simulation_object, current_candidate.content)
        if response == 1:
            # it's preferred: look at the lower rewards
            remainder_to_search = remainder_to_search[current_idx:]
        else:
            remainder_to_search = remainder_to_search[:current_idx]
    # finished process
    reward_bound = remainder_to_search[0].reward_value
    print("Reward boundary retrieved is {}".format(reward_bound))
    print("Reward weights for task are {}".format(reward_values))
    return reward_bound