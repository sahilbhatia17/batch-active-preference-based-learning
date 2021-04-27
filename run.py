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
	if task == 'driver':
		if N == 30:
			reward_weights = np.array([0.62867544, -0.49269658,  0.49139338,  0.34720284])
			reward_threshold = demos.find_threshold(N, M, reward_weights, task='driver', method="nonbatch")
		elif N == 40:
			reward_weights = np.array([0.56694832,-0.47224288 , 0.47278642 , 0.48169415])
			reward_threshold = demos.find_threshold(N, M, reward_weights, task='driver', method="nonbatch")
		elif N == 50:
			reward_weights = np.array([0.60316787, -0.51733234,  0.48007103, 0.37160137])
			reward_threshold = demos.find_threshold(N, M, reward_weights, task='driver', method="nonbatch")
		else:
			print("no reward weights, run preference query with N = ", N)
	elif task == 'mountaincar':
		reward_weights = np.array([-0.67978501, -0.56311196, 0.4698907])
		reward_threshold = demos.find_threshold(N, M, reward_weights, task='mountaincar', method="nonbatch")
	elif task == 'lunarlander':
		reward_weights = np.array([])
		reward_threshold = demos.find_threshold(N, M, reward_weights, task='lunarlander', method="nonbatch")
	else:
		print("no reward function available for non-driver task")

elif method == "comparison":
	reward_weights = np.array([0.60316787, -0.51733234, 0.48007103, 0.37160137])
	boundary = 0.90
	demos.run_comparison_plots(N, 10, reward_weights, boundary, M,
                         task='driver', method='nonbatch')
else:
    print('There is no method called ' + method)

