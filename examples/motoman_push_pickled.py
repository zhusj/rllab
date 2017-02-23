from rllab.algos.trpo import TRPO
from rllab.algos.erwr import ERWR
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.motoman_push_env import MOtomanPushEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.misc.instrument import run_experiment_lite
from rllab.sampler.utils import rollout



def run_task(*_):
	env = normalize(MOtomanPushEnv())
	# env = MOtomanPushEnv()

	print ("action_space: ", env.spec.action_space.low,env.spec.action_space.high)

	policy = GaussianGRUPolicy(
	    env_spec=env.spec,
	)

	common_batch_algo_args = dict(
	    n_itr=30,
	    batch_size=1,
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


	o = env.reset()
	policy.reset()
	best_action, agent_info = policy.get_action(o)
	print ("final policy: ", best_action, agent_info)


	path = rollout(env, policy, animated=True)


	# print ('best_action: ', best_action[0])
	# env = MOtomanPushEnv()
	# env.step(best_action[0])

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    plot=True,
    exp_prefix = "ERWR"
)

