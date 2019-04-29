
import argparse

import tensorflow as tf
import numpy as np

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

import gym_super_mario_bros

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class CNN(object):
	def __init__(self, args):
		self._args = args
		self.build_network()

	def __del__(self):
		pass

	def build_network(self):
		def conv_layer(inputs=None,
				conv_filters=8, conv_kernel_size=6, conv_strides=(1,1), conv_padding='same',
				pool_size=(3,3), pool_strides=(1,1), pool_padding='same'):
			step = tf.layers.conv2d(inputs=inputs, filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_strides, padding=conv_padding)
			step = tf.nn.relu(step)
			step = tf.layers.max_pooling2d(inputs=step, pool_size=pool_size, strides=pool_strides, padding=pool_padding)

			return step

		# input image
		step = tf.placeholder(tf.float32, shape=[None, 240, 256, 3])
		self._input = step

		# conv layers
		step = conv_layer(inputs=step,
				conv_filters=12, conv_kernel_size=4, conv_strides=(1,1), conv_padding='same',
				pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	# (?, 240, 256, 8)

		step = conv_layer(inputs=step,
				conv_filters=24, conv_kernel_size=6, conv_strides=(2,2), conv_padding='same',
				pool_size=(2,2), pool_strides=(1,1), pool_padding='same')	# (?, 120, 128, 16)	

		step = conv_layer(inputs=step,
				conv_filters=36, conv_kernel_size=8, conv_strides=(1,1), conv_padding='same',
				pool_size=(4,4), pool_strides=(2,2), pool_padding='same')	# (?, 60, 64, 24)

		step = conv_layer(inputs=step,
				conv_filters=48, conv_kernel_size=8, conv_strides=(2,2), conv_padding='same',
				pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 15, 16, 32)

		step = tf.layers.max_pooling2d(inputs=step, pool_size=(15,16), strides=(1,1), padding='valid')

		# feature vector
		step = tf.squeeze(step, [1, 2])
		self._output = step

class ReplayMemory(obejct):
	def __init__(self, args):
		pass

	def push(self, state, action, reward, next_sate):
		new_item = (state, action, reward, next_state)

def main():
	# gym env
	env = gym_super_mario_bros.make('SuperMarioBros-v0')
	env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

	# generate args
	parser = argparse.ArgumentParser(description="SuperMarioBros")
	parser.add_argument('--action_count', default=env.action_space.n, type=int, help="The number of available actions")
	parser.add_argument('--replay_memory_total', default=100000, type=int, help="The number of available actions")
	args = parser.parse_args()

	args.action_count

	# generate tf graph
	tf.reset_default_graph()

	cnn = CNN(args)

	done = True
	for step in range(1):
		if done:
			state = env.reset()
		state, reward, done, info = env.step(env.action_space.sample())

	env.close()

if __name__ == '__main__':
	main()
