import demos
import sys
import numpy as np
#import automata_learning

task   = sys.argv[1].lower()
method = sys.argv[2].lower()
N = int(sys.argv[3])
M = int(sys.argv[4])

if method == 'nonbatch' or method == 'random':
    #arb weights just to see if sampler would update weights with nonbatch        
    weights = [-0.2338154,   0.62484734, -0.74491353]
    demos.nonbatch(task, method, N, M, weights)
elif method == 'greedy' or method == 'medoids' or method == 'boundary_medoids' or method == 'successive_elimination':
    b = int(sys.argv[5])
    demos.batch(task, method, N, M, b)
elif method == "threshold":
    num_samples = int(sys.argv[5])
    if task == 'driver':
        if N == 30:
            reward_weights = np.array([0.62867544, -0.49269658,  0.49139338,  0.34720284])
            gt_threshold = 0.84
            reward_threshold = demos.find_threshold(num_samples, M, reward_weights, gt_threshold, task='driver', method="nonbatch")
        elif N == 40:
            reward_weights = np.array([0.56694832,-0.47224288 , 0.47278642 , 0.48169415])
            gt_threshold = 0.84
            reward_threshold = demos.find_threshold(num_samples, M, reward_weights, gt_threshold, task='driver', method="nonbatch")
        elif N == 50:
            reward_weights = np.array([0.60316787, -0.51733234,  0.48007103, 0.37160137])
            gt_threshold = 0.84
            reward_threshold = demos.find_threshold(num_samples, M, reward_weights, gt_threshold, task='driver', method="nonbatch")
        else:
            print("no reward weights, run preference query with N = ", N)
    elif task == 'mountaincar':
        reward_weights = np.array([-0.67978501, -0.56311196, 0.4698907])
        gt_threshold = 0.84
        reward_threshold = demos.find_threshold(num_samples, M, reward_weights, gt_threshold, task='mountaincar', method="nonbatch")
    elif task == 'lunarlander':
        reward_weights = np.array([])
        gt_threshold = 0.84
        reward_threshold = demos.find_threshold(num_samples, M, reward_weights, gt_threshold, task='lunarlander', method="nonbatch")
    elif task == "circles":
        reward_weights = np.array( [-0.2259573,  0.59799658, -0.76898855])
        gt_threshold = 0.023990068857129017
        reward_threshold = demos.find_threshold(num_samples, M, reward_weights, gt_threshold, task='circles', method='nonbatch')
    else:
        print("no reward function available for non-driver task")

elif method == "comparison":
    reward_weights = np.array([0.56687795 ,-0.51010378  ,0.5178173 , 0.38769675])
    boundary = 0.70
    learned_reward_list = [[0.52239328, -0.57670694,  0.61716699,  0.11670168],[0.59893698, -0.47751113,  0.61883226,  0.17408115],
                           [0.62867544, -0.49269658,  0.49139338,  0.34720284], [0.56694832,-0.47224288 , 0.47278642 , 0.48169415],
                           [0.60316787, -0.51733234,  0.48007103, 0.37160137], [0.56524005, -0.51794572, 0.51361789,  0.3852695],
                           [0.59673816, -0.51312493, 0.48763488,  0.37791347]]
    demos.run_preference_comparison([10, 20, 30, 40, 50, 60], 10, reward_weights, boundary, M,
                         task='driver', method='nonbatch', provided_weights=learned_reward_list)

else:
    print('There is no method called ' + method)

