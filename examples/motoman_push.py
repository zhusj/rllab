from rllab.algos.trpo import TRPO
from rllab.algos.erwr import ERWR
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.motoman_push_env import MOtomanPushEnv
from rllab.envs.motoman_push_env_real import MOtomanPushEnvReal

from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.sampler.utils import rollout

import math

total_iter = 10
num_of_trials = 5

for trial in range(0,num_of_trials):   
	for iter in range(0,total_iter):
		print ('Testing trial: ', trial, 'iteration: ', iter)

		env = normalize(MOtomanPushEnvReal())
		# env = MOtomanPushEnv()

		print ("action_space: ", env.spec.action_space.low,env.spec.action_space.high)

		policy = GaussianGRUPolicy(
		    env_spec=env.spec,
		)

		common_batch_algo_args = dict(
		    n_itr=iter+1,
		    batch_size=5,
		    max_path_length=100,
		)

		baseline = LinearFeatureBaseline(env_spec=env.spec)
		algo = ERWR(
		    env=env,
		    policy=policy,
		    baseline=baseline,
		    **common_batch_algo_args
		)
		algo.train()


		# o = env.reset()
		# policy.reset()
		# best_action, agent_info = policy.get_action(o)
		# print ("final policy: ", best_action, agent_info)


		# path = rollout(env, policy)

		# reward=path['rewards'][0]
		# dist = math.log(reward)/(-2)
		# print ('path: ', path)
		# print ('rewards: ', reward)
		# print ('dist: ', dist)



		input("Press Enter to start training...")
