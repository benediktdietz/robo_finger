import sys, os, argparse, logging, chainer, chainerrl
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
import gym
import gym.wrappers
import numpy as np
from matplotlib import pyplot as plt

from learning_finger_manipulation import envs


parser = argparse.ArgumentParser()
parser.add_argument('--outdir',             	type=str, default='run_logs',   help='Directory path to save output files. it will be created if not existent.')
parser.add_argument('--gpu',                	type=int, default=-1,           help='GPU to use, set to -1 if no GPU.')
parser.add_argument('--load',               	type=str, default='',           help='Directory to load agent from.')
parser.add_argument('--steps',              	type=int, default=10 ** 5,      help='Total number of timesteps to train the agent.')
parser.add_argument('--trpo-update-interval',	type=int, default=5000,			help='Interval steps of TRPO iterations.')
parser.add_argument('--eval-n-runs',        	type=int, default=10,           help='Number of episodes run for each evaluation.')
parser.add_argument('--eval-interval',      	type=int, default=100,          help='Interval in timesteps between evaluations.')
parser.add_argument('--replay-start-size',  	type=int, default=1000,         help='Minimum replay buffer size before performing gradient updates.')
parser.add_argument('--batch-size',         	type=int, default=64,           help='Minibatch size')
parser.add_argument('--logger-level',       	type=int, default=logging.INFO, help='Level of the root logger.')
parser.add_argument('--render',             	action='store_true',            help='Render env states in a GUI window.')
parser.add_argument('--demo',               	action='store_true',            help='Just run evaluation, not training.')
parser.add_argument('--monitor',            	action='store_true',            help='Wrap env with gym.wrappers.Monitor.')
args = parser.parse_args()


logging.basicConfig(level=logging.INFO)


def make_env(test=False, print_out=False):

	env = gym.make(
		"DaktyPushingSimulationEnv-v0",
		level=5,
		simulation_backend="mujoco",
		control_frequency_in_hertz=100,
		state_space_components_to_be_used=None,
		alternate_env_object=None,
		discretization_factor_torque_control_space=None,
		model_as_function_for_pixel_to_latent_space_parsing=(None, None))


	env.unwrapped.finger.set_resolution_quality('low')

	if print_out: print('\n############\n', env, '\n############\n')

	env = gym.wrappers.TimeLimit(env)

	if print_out: print('\n############\n', env, '\n############\n')


	# Unwrap TimeLimit wrapper
	assert isinstance(env, gym.wrappers.TimeLimit)
	env = env.env

	# Use different random seeds for train and test envs
	env_seed = 421 if test else 420
	env.seed(env_seed)

	# Cast observations to float32
	env = chainerrl.wrappers.CastObservationToFloat32(env)

	if args.monitor: env = chainerrl.wrappers.Monitor(env, args.outdir)
	if args.render and not test: env = chainerrl.wrappers.Render(env)

	return env


env = make_env()


obs_space = env.observation_space
action_space = env.action_space

timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

print('observation space:', obs_space.shape)
print('action space:', action_space)
print('timestep_limit:', timestep_limit)

print(obs_space.low.size)

print('\n\n\n\n')


obs_normalizer = chainerrl.links.EmpiricalNormalization(obs_space.low.size)

# Use a value function to reduce variance
value_function = chainerrl.v_functions.FCVFunction(
	obs_space.low.size,
	n_hidden_channels=64,
	n_hidden_layers=2,
	last_wscale=0.01,
	nonlinearity=F.tanh
	)

vf_opt = chainer.optimizers.Adam()
vf_opt.setup(value_function)


policy = chainerrl.policies.FCGaussianPolicyWithStateIndependentCovariance(
	obs_space.low.size,
	action_space.low.size,
	n_hidden_channels=64,
	n_hidden_layers=2,
	mean_wscale=0.01,
	nonlinearity=F.tanh,
	var_type='diagonal',
	var_func=lambda x: F.exp(2 * x),  # Parameterize log std
	var_param_init=0  # log std = 0 => std = 1
	)


agent = chainerrl.agents.TRPO(
	policy=policy,
	vf=value_function,
	vf_optimizer=vf_opt,
	obs_normalizer=obs_normalizer,
	update_interval=args.trpo_update_interval,
	conjugate_gradient_max_iter=20,
	conjugate_gradient_damping=1e-1,
	gamma=0.995,
	lambd=0.97,
	vf_epochs=5,
	entropy_coef=0
	)








n_episodes = 10000

max_episode_len = 10000
for i in range(1, n_episodes + 1):
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while not done and t < max_episode_len:
        # Uncomment to watch the behaviour
        # env.render()
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
    if i % 10 == 0:
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
    agent.stop_episode_and_train(obs, reward, done)
print('Finished.')









