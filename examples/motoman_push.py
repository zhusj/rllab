from rllab.algos.trpo import TRPO
from rllab.algos.erwr import ERWR
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.motoman_push_env import MOtomanPushEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy


env = normalize(MOtomanPushEnv())
policy = GaussianGRUPolicy(
    env_spec=env.spec,
)

# common_batch_algo_args = dict(
#     n_itr=10,
#     batch_size=1,
#     max_path_length=100,
# )

baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = ERWR(
    env=env,
    policy=policy,
    baseline=baseline,
    # **common_batch_algo_args
)
algo.train()

print ("final policy: ", policy)