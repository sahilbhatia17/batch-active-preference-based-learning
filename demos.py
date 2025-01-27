from sampling import Sampler
import algos
import numpy as np
from simulation_utils import create_env, get_feedback, run_algo, get_membership_feedback, get_feedback_auto,get_membership_feedback_auto
import sys
import math
import lattice
from sklearn import svm
import pickle

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



def nonbatch(task, method, N, M, checkpoints=None):
    if checkpoints is None:
        checkpoints = []
    checkpointed_weights = []
    simulation_object = create_env(task)
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []
    input_A = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(2*simulation_object.feed_size))
    input_B = np.random.uniform(low=2*lower_input_bound, high=2*upper_input_bound, size=(2*simulation_object.feed_size))
    psi, s = get_feedback_auto(simulation_object, input_A, input_B) # psi is the difference, s is the 1 or -1 signal
    psi_set.append(psi)
    s_set.append(s)
    for i in range(1, N):
        w_sampler.A = psi_set
        w_sampler.y = np.array(s_set).reshape(-1,1)
        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples,axis=0)
        print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
        if i in checkpoints:
            checkpointed_weights.append(mean_w_samples/np.linalg.norm(mean_w_samples))
            print("Weights saved at iteration {}".format(i))
        input_A, input_B = run_algo(method, simulation_object, w_samples)
        psi, s = get_feedback_auto(simulation_object, input_A, input_B)
        psi_set.append(psi)
        s_set.append(s)
    w_sampler.A = psi_set
    w_sampler.y = np.array(s_set).reshape(-1,1)
    w_samples = w_sampler.sample(M)
    checkpointed_weights.append(mean_w_samples/np.linalg.norm(mean_w_samples))
    print('w-estimate = {}'.format(mean_w_samples/np.linalg.norm(mean_w_samples)))
    return checkpointed_weights

def run_comparison_plots(num_preference_queries, num_membership_queries, ground_truth_reward, ground_truth_boundary, M, task='driver', method='nonbatch'):
    bsearch_num_samples = 2 ** num_membership_queries
    preference_learned_rewards = nonbatch(task, method, num_preference_queries, M, checkpoints=[num_preference_queries])[-1]
    simulation_object = create_env(task)
    # collect trajectories
    reward_samples_full_set = collect_trajectories(simulation_object, method, bsearch_num_samples, preference_learned_rewards)
    random_samples_full_set = collect_trajectories(simulation_object, "random", num_membership_queries, preference_learned_rewards)

    # get the boundary and SVM values from the query methods
    preference_bsearch_boundary = membership_threshold(lattice.sort_on_rewards(reward_samples_full_set), simulation_object, get_labels=False)
    preference_svm_coeff, preference_svm_boundary, preference_svm = svm_threshold(reward_samples_full_set[:num_membership_queries], simulation_object)
    random_svm_coeff, random_svm_boundary, random_svm = svm_threshold(random_samples_full_set, simulation_object)
    # normalize the preference coefficients
    preference_svm_boundary = preference_svm_boundary / np.linalg.norm(preference_svm_coeff)
    preference_svm_coeff = preference_svm_coeff / np.linalg.norm(preference_svm_coeff)
    random_svm_boundary = random_svm_boundary / np.linalg.norm(random_svm_coeff)
    random_svm_coeff = random_svm_coeff / np.linalg.norm(random_svm_coeff)

    #now, use them for evaluation
    def compute_angle(vec1, vec2):
        return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
    difference_values = {}
    difference_values["svm w/ reward"] = compute_angle(preference_svm_coeff, ground_truth_reward) \
                                         + abs(preference_svm_boundary - ground_truth_boundary)

    difference_values["svm w/ random"] = compute_angle(random_svm_coeff, ground_truth_reward) \
                                         + abs(random_svm_boundary - ground_truth_boundary)
    difference_values["bsearch w/ reward"] = compute_angle(preference_learned_rewards, ground_truth_reward) \
                                         + abs(preference_bsearch_boundary - ground_truth_boundary)
    print("Differences computed are: {}".format(difference_values))
    return difference_values


def collect_trajectories(simulation_object, samplemethod, num_samples, reward_values):
    trajectory_set = []
    def add_traj(samplemethod, traj_set):

        sample_A, sample_B = run_algo(samplemethod, simulation_object, reward_values.reshape(1,-1))
        simulation_object.feed(sample_A)
        phi_A = simulation_object.get_features()
        # now, compute the reward for each sample
        reward_A = np.sum(reward_values * phi_A)
        traj_set.append(lattice.Node(sample_A, reward_value=reward_A, features=phi_A))
    for idx in range(num_samples):
        add_traj(samplemethod, trajectory_set)
    return trajectory_set

def find_threshold(num_weighted_samples, num_random_samples, reward_values, num_membership_queries=0, task='driver', method="nonbatch"):
    # first, sample the trajectories from the distribution\
    simulation_object = create_env(task)
    d = simulation_object.num_of_features
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    # set the reward weights of the sampler
    # set the number of membership queries as a log function of the total # of samples
    num_membership_queries = max(num_membership_queries, int(math.ceil(math.log(num_weighted_samples + num_random_samples))))
    reward_traj_set = collect_trajectories(simulation_object, method, num_weighted_samples, reward_values)
    random_traj_set = collect_trajectories(simulation_object, "random", num_random_samples, reward_values)
    w_true = np.array([ 0.56687795 ,-0.51010378  ,0.5178173 ,  0.38769675])


    svm_reward_set = reward_traj_set[:num_membership_queries] + collect_trajectories(simulation_object, method, num_weighted_samples, reward_values)
    #adding n more samples to the svm dataset --> n + log(n) samples
    
    svm_random_set = random_traj_set[:num_membership_queries] + collect_trajectories(simulation_object, "random", num_weighted_samples, reward_values)
    #adding n more samples to the svm dataset --> n + log(n) samples

    
    full_traj_set = reward_traj_set + random_traj_set
    # sort the trajectories by reward
    sorted_lattice = lattice.sort_on_rewards(full_traj_set)

    #test set trajectories sampled from w_true
    f_reward = open('reward_test_set.obj','rb')
    reward_traj_set_test = pickle.load(f_reward)
    f_reward.close()
    #test set trajectories sampled randomly
    f_random = open('random_test_set.obj','rb')
    random_traj_set_test = pickle.load(f_random)
    f_random.close()

    #get data and labels for the test set
    x = []
    y = []
    r = []
    reward_traj_set_test = reward_traj_set_test  +  random_traj_set_test
    for node in reward_traj_set_test:
        #print(node.reward_value)
        x.append(node.features)
        reward = np.sum(np.dot(w_true,node.features))
        r.append(reward)
        if reward < 0.74:
            y.append(0)
        else:
            y.append(1)
    print(y)

    #now, begin getting membership query feedback on things
    bsearch_reward_bound, labeled_data = membership_threshold(sorted_lattice, simulation_object, get_labels=True)
    svm_bsearch_coeff, svm_bsearch_inter, clssfr_bsearch = svm_threshold(svm_reward_set, simulation_object, labeled_samples=labeled_data)
    svm_reward_coeff, svm_reward_inter, clssfr_reward = svm_threshold(svm_reward_set, simulation_object)
    svm_random_coeff, svm_random_inter, clssfr_random = svm_threshold(svm_random_set, simulation_object)
    # finished process
    print("Reward boundary retrieved from binary search method is {}".format(bsearch_reward_bound))
    print("SVM coefficient and intercept for same queries as binary search are: {} and {}".format(svm_bsearch_coeff, svm_bsearch_inter))
    print("SVM coefficient and intercept for reward-sampled queries are: {} and {}".format(svm_reward_coeff, svm_reward_inter))
    print("SVM coefficient and intercept for random-sampled queries are: {} and {}".format(svm_random_coeff, svm_random_inter))
    print("Reward weights for task are {}".format(reward_values))

    acc_bsearch = get_accuracy(r,y,reward_bound=bsearch_reward_bound,clssfr=None)
    acc_svm_learnt = get_accuracy(x,y,reward_bound=None,clssfr=clssfr_reward)
    acc_svm_random = get_accuracy(x,y,reward_bound=None,clssfr=clssfr_random)

    print("Accuracy for binary search is ", acc_bsearch)
    print("Accuracy for svm with reward-sampled queries is ", acc_svm_learnt)
    print("Accuracy for svm with randomly-sampled queries is ", acc_svm_random)

def get_accuracy(x,y,reward_bound=None,clssfr=None):
    if clssfr is not None:
        predictions  = clssfr.predict(x)
        correct = 0
        for idx,i in enumerate(predictions):
            if i == y[idx]:
                correct += 1
        return correct/len(x)
    else:
        predicted_bsearch = []
        for reward in x:
            if reward < reward_bound:
                predicted_bsearch.append(0)
            else:
                predicted_bsearch.append(1)
        correct = 0
        for idx,i in enumerate(predicted_bsearch):
            if i == y[idx]:
                correct += 1
        return correct/len(x)


def membership_threshold(sorted_lattice, simulation_object, get_labels=True):
    #now, begin getting membership query feedback on things
    remainder_to_search = sorted_lattice
    xs, ys = [], []
    while len(remainder_to_search) > 1:
        current_idx = int(len(remainder_to_search) / 2)
        current_candidate = remainder_to_search[current_idx]
        # ask: is this a member of the preferred set?
        print("current reward boundary is: {}".format(current_candidate.reward_value))
        response = get_membership_feedback_auto(simulation_object, current_candidate.content)
        if response == 1:
            # it's preferred: look at the lower rewards
            remainder_to_search = remainder_to_search[current_idx:]
        else:
            remainder_to_search = remainder_to_search[:current_idx]
        if get_labels:
            xs.append(current_candidate.features)
            ys.append(max(0, response))
    # finished process
    if get_labels:
        return remainder_to_search[0].reward_value, (xs, ys)
    return remainder_to_search[0].reward_value

def svm_threshold(sampled_nodes, simulation_object, labeled_samples=None):
    # get membership query feedback on samples to collect positive and negative samples
    # treat the feature averages as the
    print("Beginning SVM membership query method.")
    Xs = []
    Ys = []
    if labeled_samples is None:
        # ask membership queries on provided samples
        for node in sampled_nodes:
            response = get_membership_feedback_auto(simulation_object, node.content)
            Xs.append(node.features)
            Ys.append(max(0, response))  # will be 1 for positive, 0 for negative, for SVM purposes
    else:
        Xs, Ys = labeled_samples
    # fit an SVM to the labeled samples
    clssfr = svm.LinearSVC()
    clssfr.fit(Xs, Ys)
    return clssfr.coef_, clssfr.intercept_,clssfr

