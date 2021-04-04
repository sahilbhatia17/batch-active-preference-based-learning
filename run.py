import demos
import sys
import numpy as np

task   = sys.argv[1].lower()
method = sys.argv[2].lower()
N = int(sys.argv[3])
M = int(sys.argv[4])

if method == 'nonbatch' or method == 'random':
    demos.nonbatch(task, method, N, M)
elif method == 'greedy' or method == 'medoids' or method == 'boundary_medoids' or method == 'successive_elimination':
    b = int(sys.argv[5])
    demos.batch(task, method, N, M, b)
elif method == "threshold":
    if task != 'driver':
        print("no reward function available for non-driver task")
    else:
        reward_weights = np.array([ 0.56687795 ,-0.51010378  ,0.5178173 ,  0.38769675])
        reward_threshold = demos.membership_threshold(N, M, reward_weights, task='driver', method="nonbatch")
else:
    print('There is no method called ' + method)

