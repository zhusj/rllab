import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

from rllab.misc import logger
import os

import gym
import gym.wrappers
import gym.envs
import gym.spaces
import traceback
import logging

try:
    from gym.wrappers.monitoring import logger as monitor_logger

    monitor_logger.setLevel(logging.WARNING)
except Exception as e:
    traceback.print_exc()


class CappedCubicVideoSchedule(object):
    # Copied from gym, since this method is frequently moved around
    def __call__(self, count):
        if count < 1000:
            return int(round(count ** (1. / 3))) ** 3 == count
        else:
            return count % 1000 == 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']

        video_schedule = CappedCubicVideoSchedule()
        log_dir = './videos/'
        env.wrapped_env.env.env = gym.wrappers.Monitor(env.wrapped_env.env.env, log_dir, video_callable=video_schedule, force=True)
        env.monitoring = True


        while True:
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=True, speedup=args.speedup, always_return_paths=True)
            print("rewards: ", sum(path["rewards"]))
            if not query_yes_no('Continue simulation?'):
                break
