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

from rllab.misc.instrument import run_experiment_lite

def run_task(*_):
    env = normalize(MOtomanPushEnvReal())
    # env = MOtomanPushEnv()

    print ("action_space: ", env.spec.action_space.low,env.spec.action_space.high)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
#         hidden_sizes=(16,32,32,16),
#         std_hidden_sizes=(16,32,32,16),
    )

    common_batch_algo_args = dict(
        n_itr=10,
        batch_size=5,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = ERWR(
        env=env,
        policy=policy,
        baseline=baseline,
#         step_size=0.001,
#         gae_lambda=0.5,
        **common_batch_algo_args
    )
    algo.train()

    path = rollout(env, policy)
    reward=path['rewards'][0]
    dist = math.log(reward)/(-2)

    time_result[trial, iter] = elapsed_time
    reward_result[trial, iter] = reward
    dist_result[trial, iter] = dist

    print ('path: ', path)
    print ('reward: ', reward)
    print ('dist: ', dist)

    
    
    
num_of_trials = 1

for trial in range(0,num_of_trials):    
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        # n_parallel=0,
        exp_prefix = "ERWR"
    )