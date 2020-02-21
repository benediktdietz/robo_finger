"""A training script of TD3 on OpenAI Gym Mujoco environments.

This script follows the settings of http://arxiv.org/abs/1802.09477 as much
as possible.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
import logging
import os
import sys

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import gym
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import replay_buffer

from learning_finger_manipulation import envs

from tqdm import tqdm


def make_env(test):

    render = True

    env = gym.make(
        "DaktyPushingSimulationEnv-v0",
        level=5,
        simulation_backend="mujoco",
        control_frequency_in_hertz=100,
        state_space_components_to_be_used=None,
        alternate_env_object=None,
        discretization_factor_torque_control_space=None,
        model_as_function_for_pixel_to_latent_space_parsing=(None, None)
        )

    print('\n############\n', env, '\n############\n')

    env.unwrapped.finger.set_resolution_quality('low')

    print('\n############\n', env, '\n############\n')

    env = gym.wrappers.TimeLimit(env)

    print('\n############\n', env, '\n############\n')


    # Unwrap TimeLimit wrapper
    assert isinstance(env, gym.wrappers.TimeLimit)
    env = env.env

    # Use different random seeds for train and test envs

    env = chainerrl.wrappers.CastObservationToFloat32(env)
   
    if render and not test:
        env = chainerrl.wrappers.Render(env)
    return env



env = make_env(test=True)

obs_space = env.observation_space
action_space = env.action_space
print('Observation space:', obs_space)
print('Action space:', action_space)

action_size = action_space.low.size


sys.stdout.flush()

print('\nbeginning training\n')





obs = env.reset()


for i in range(100000):

    print(i, end='\r')

    env.render()







    # eval_env = make_env(test=True)
    # if args.demo:
    #     eval_stats = experiments.eval_performance(
    #         env=eval_env,
    #         agent=agent,
    #         n_steps=None,
    #         n_episodes=args.eval_n_runs,
    #         max_episode_len=timestep_limit)
    #     print('n_runs: {} mean: {} median: {} stdev {}'.format(
    #         args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
    #         eval_stats['stdev']))
    # else:
    #     experiments.train_agent_with_evaluation(
    #         agent=agent, 
    #         env=env, 
    #         steps=args.steps,
    #         eval_env=eval_env, 
    #         eval_n_steps=None,
    #         eval_n_episodes=args.eval_n_runs, 
    #         eval_interval=args.eval_interval,
    #         outdir=args.outdir,
    #         train_max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()

