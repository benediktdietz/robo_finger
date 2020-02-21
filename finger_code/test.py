import gym
import matplotlib.pyplot as plt
from learning_finger_manipulation import envs

# import ray
# ray.init()

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

assert isinstance(env, gym.wrappers.TimeLimit)

env.unwrapped.finger.set_resolution_quality('high')

image = env.unwrapped.render('rgb_array')

print(image.shape)