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




def concat_obs_and_action(obs, action, print_dims=False):
    """Concat observation and action to feed the critic."""

    # print('init obs.shape..........', obs.shape)
    # print('init action.shape.......', action.shape, '\n')

    img_conv_1 = L.Convolution2D(3, 4, ksize=3, stride=1, pad=0)
    img_conv_2 = L.Convolution2D(4, 8, ksize=3, stride=1, pad=0)

    img_lin_1 = L.Linear(64)

    batchnorm = L.BatchNormalization(axis=(1,2,3))

    obs = F.reshape(obs, (-1, 3, obs.shape[1], obs.shape[2]))

    # if obs.shape[0] == 1: 
    #     print_dims = True
    # else: 
    #     print_dims = False

    if print_dims: 
        print('\n\n### batchsize x channels x height x width ###')
        print('shape in...............', obs.shape)


    obs = F.max_pooling_2d(obs, 2)
    if print_dims: print('after pool0............', obs.shape)

    obs = batchnorm(obs)
    if print_dims: print('after BN1..............', obs.shape)

    obs = img_conv_1(obs)
    if print_dims: print('after conv1............', obs.shape)

    obs = F.max_pooling_2d(obs, 2)
    if print_dims: print('after pool1............', obs.shape)


    obs = batchnorm(obs)
    if print_dims: print('after BN2..............', obs.shape)

    obs = img_conv_2(obs)
    if print_dims: print('after conv2............', obs.shape)

    obs = F.max_pooling_2d(obs, 2)
    if print_dims: print('after pool2............', obs.shape)



    obs = F.reshape(obs, (action.shape[0], -1))
    if print_dims: print('after reshape..........', obs.shape)

    obs = img_lin_1(obs)

    if print_dims: 
        print('after lin1.............', obs.shape)
        print('passed on..............', F.concat((obs, action), axis=-1).shape, '\n\n\n')



    return F.concat((obs, action), axis=-1)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env',                type=str, default='Inv',        help='OpenAI Gym MuJoCo env to perform algorithm on.')
    parser.add_argument('--outdir',             type=str, default='results',    help='Directory path to save output files. it will be created if not existent.')
    parser.add_argument('--seed',               type=int, default=420,          help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu',                type=int, default=-1,           help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--load',               type=str, default='',           help='Directory to load agent from.')
    parser.add_argument('--steps',              type=int, default=10 ** 5,      help='Total number of timesteps to train the agent.')
    parser.add_argument('--eval-n-runs',        type=int, default=10,           help='Number of episodes run for each evaluation.')
    parser.add_argument('--eval-interval',      type=int, default=100,          help='Interval in timesteps between evaluations.')
    parser.add_argument('--replay-start-size',  type=int, default=1000,         help='Minimum replay buffer size before performing gradient updates.')
    parser.add_argument('--batch-size',         type=int, default=4,            help='Minibatch size')
    parser.add_argument('--logger-level',       type=int, default=logging.INFO, help='Level of the root logger.')
    parser.add_argument('--render',             action='store_true',            help='Render env states in a GUI window.')
    parser.add_argument('--demo',               action='store_true',            help='Just run evaluation, not training.')
    parser.add_argument('--monitor',            action='store_true',            help='Wrap env with gym.wrappers.Monitor.')
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    def make_env(test):

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
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(env, args.outdir)
        if args.render and not test:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = env.observation_space
    action_space = env.action_space
    print('Observation space:', obs_space)
    print('Action space:', action_space)

    action_size = action_space.low.size

    winit = chainer.initializers.LeCunUniform(3 ** -0.5)

    '''
    define policy and optimiser
    output_dim = action_size
    '''
    policy = chainer.Sequential(
        L.Linear(None, 128, initialW=winit),
        F.relu,
        L.Linear(None, 64, initialW=winit),
        F.relu,
        L.Linear(None, action_size, initialW=winit),
        F.tanh,
        chainerrl.distribution.ContinuousDeterministicDistribution,
    )
    policy_optimizer = optimizers.Adam(3e-4).setup(policy)

    # policy.to_gpu(0)

    '''
    define q-function and optimiser
    output_dim = 1
    defines 2 identical q_functions with resp. optimisers
    '''
    def make_q_func_with_optimizer():
        q_func = chainer.Sequential(
            concat_obs_and_action,
            L.Linear(None, 128, initialW=winit),
            F.relu,
            L.Linear(None, 64, initialW=winit),
            F.relu,
            L.Linear(None, 1, initialW=winit),
        )
        q_func_optimizer = optimizers.Adam().setup(q_func)
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    # q_func1.to_gpu(0)
    # q_func2.to_gpu(0)


    print('\n\n-------------------\n', obs_space.low.shape, '\n-------------------\n')

    # Draw the computational graph and save it in the output directory.
    fake_obs = chainer.Variable(
        policy.xp.zeros_like(obs_space.low, dtype=np.float32)[None],
        name='observation')
    fake_action = chainer.Variable(
        policy.xp.zeros_like(action_space.low, dtype=np.float32)[None],
        name='action')
    chainerrl.misc.draw_computational_graph(
        [policy(fake_obs)], os.path.join(args.outdir, 'policy'))
    chainerrl.misc.draw_computational_graph(
        [q_func1(fake_obs, fake_action)], os.path.join(args.outdir, 'q_func1'))
    chainerrl.misc.draw_computational_graph(
        [q_func2(fake_obs, fake_action)], os.path.join(args.outdir, 'q_func2'))

    rbuf = replay_buffer.ReplayBuffer(10 ** 5)

    explorer = explorers.AdditiveGaussian(
        scale=0.1, low=action_space.low, high=action_space.high)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = chainerrl.agents.TD3(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        soft_update_tau=5e-3,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
    )

    # agent.to_gpu(0)

    if len(args.load) > 0:
        agent.load(args.load)


    sys.stdout.flush()

    print('\nbeginning training\n')


    n_episodes = 10000

    # pbar = tqdm(total=n_episodes)

    max_episode_len = 5000
    for i in range(1, n_episodes + 1):

        # pbar.update(1)


        obs = env.reset()
        # print('obs inital..............', obs.shape)
        reward = 0
        done = False
        R = 0  # return (sum of rewards)
        t = 0  # time step

        pbar = tqdm(total=max_episode_len)

        while not done and t < max_episode_len:

            pbar.update(1)


            # Uncomment to watch the behaviour
            # env.render()
            action = agent.act_and_train(obs, reward)
            # print('action..................', action)

            obs, reward, done, _ = env.step(action)
            # print('obs.....................', obs)
            # print('reward..................', reward)

            R += reward
            t += 1

        if i % 1 == 0:
            print('episode:', i,
                  'R:', R,
                  'statistics:', agent.get_statistics())
        agent.stop_episode_and_train(obs, reward, done)
    print('Finished.')





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

