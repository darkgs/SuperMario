
import random
import argparse

import tensorflow as tf
import numpy as np

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

import gym_super_mario_bros

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


class ReplayMemory(object):
	def __init__(self, args):
		self._args = args
		self._mem = []

	def __len__(self):
		return len(self._mem)

	def push(self, state, action, reward, next_state, done):
		new_item = (state, action, reward, next_state, 1.0 if done else 0.0)
		self._mem.append(new_item)

		if len(self._mem) > self._args.replay_memory_total:
			self._mem = self._mem[-self._args.replay_memory_total : ]

	def get_replays(self, replay_count=None):
		if replay_count == None:
			replay_count = self._args.training_batch_size

		assert(replay_count <= len(self))

		replays = (np.random.permutation(self._mem))[:replay_count,]

		b_state = np.stack(replays[:,0].tolist())
		b_action = replays[:,1]
		b_reward = replays[:,2]
		b_next_state = np.stack(replays[:,3].tolist())
		b_done = replays[:,4]

		return b_state, b_action, b_reward, b_next_state, b_done


class DQN(object):
	def __init__(self, args):
		self._args = args
		self._dict_pred_q = self.build_q_network('pred')
		self._dict_target_q = self.build_q_network('target')

		self._copy_ops = self.build_copy_ops('pred', 'target')
		
		self._dict_optimizer = self.build_optimizer(self._dict_pred_q, self._dict_target_q)

	def train(self, sess, b_state, b_action, b_reward, b_next_state, b_done):
		feed_dict = {
			self._dict_optimizer['state']: b_state,
			self._dict_optimizer['action']: b_action,
			self._dict_optimizer['reward']: b_reward,
			self._dict_optimizer['next_state']: b_next_state,
			self._dict_optimizer['done']: b_done,
		}
		loss, _ = sess.run([self._dict_optimizer['loss'], self._dict_optimizer['optimizer']], feed_dict=feed_dict)

		return loss

	def next_action(self, sess, state):
		b_state = np.expand_dims(state, axis=0)

		feed_dict = {
			self._dict_pred_q['inputs']: b_state,
		}
		actions, = sess.run([self._dict_pred_q['selected_actions']], feed_dict=feed_dict)

		return np.squeeze(actions, axis=0).item()

	def update_target(self, sess):
		sess.run([self._copy_ops])

	def build_optimizer(self, dict_pred_q=None, dict_target_q=None):
		assert(dict_pred_q!=None and dict_target_q!=None)

		dict_optimizer = {}

		action = tf.placeholder(tf.int32, shape=(None, ))
		reward = tf.placeholder(tf.float32, shape=(None, ))
		done = tf.placeholder(tf.float32, shape=(None, ))

		# q(s, a)
		pred_q = tf.reduce_sum(dict_pred_q['outputs'] * tf.one_hot(action, self._args.action_dim, 1.0, 0.0), axis=1)
		# max_a'(Q(s', a'))
		max_target_q = tf.reduce_max(dict_target_q['outputs'], axis=1)

		y = reward + (1.0 - done) * self._args.gamma * max_target_q
		loss = tf.reduce_mean(tf.square(pred_q - tf.stop_gradient(y)))
		optimizer = tf.train.AdamOptimizer(self._args.learning_rate).minimize(loss)

		dict_optimizer['state'] = dict_pred_q['inputs']
		dict_optimizer['action'] = action
		dict_optimizer['reward'] = reward
		dict_optimizer['next_state'] = dict_target_q['inputs']
		dict_optimizer['done'] = done

		dict_optimizer['loss'] = loss
		dict_optimizer['optimizer'] = optimizer

		return dict_optimizer

	def build_q_network(self, name):
		dict_outputs = {}
		with tf.variable_scope(name):
			action_dim = self._args.action_dim

			# CNN networks
			def conv_layer(inputs=None,
					conv_filters=8, conv_kernel_size=6, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same'):
				step = tf.layers.conv2d(inputs=inputs, filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_strides, padding=conv_padding)
				step = tf.nn.relu(step)
				step = tf.layers.max_pooling2d(inputs=step, pool_size=pool_size, strides=pool_strides, padding=pool_padding)

				return step

			# input image
			step = tf.placeholder(tf.float32, shape=[None, 240, 256, 3])
			inputs = step

			# conv layers
			step = conv_layer(inputs=step,
					conv_filters=24, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 240, 256, 24)

			step = conv_layer(inputs=step,
					conv_filters=32, conv_kernel_size=3, conv_strides=(2,2), conv_padding='same',
					pool_size=(2,2), pool_strides=(1,1), pool_padding='same')	# (?, 120, 128, 32)	

			step = conv_layer(inputs=step,
					conv_filters=48, conv_kernel_size=5, conv_strides=(1,1), conv_padding='same',
					pool_size=(4,4), pool_strides=(2,2), pool_padding='same')	# (?, 60, 64, 48)

			step = conv_layer(inputs=step,
					conv_filters=48, conv_kernel_size=3, conv_strides=(2,2), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 15, 16, 48)

			step = conv_layer(inputs=step,
					conv_filters=72, conv_kernel_size=5, conv_strides=(1,1), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 8, 8, 72)

#			step = conv_layer(inputs=step,
#					conv_filters=92, conv_kernel_size=5, conv_strides=(2,2), conv_padding='same',
#					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 2, 2, 96)
#
#			step = conv_layer(inputs=step,
#					conv_filters=128, conv_kernel_size=5, conv_strides=(2,2), conv_padding='same',
#					pool_size=(2,2), pool_strides=(1,1), pool_padding='same')	# (?, 1, 1, 128)

			step = tf.reshape(step, [-1, 8*8*72])

			# feature vector
#step = tf.squeeze(step, [1, 2])

			# MLP
			step = tf.layers.dense(step, 48, activation=tf.nn.relu)
			step = tf.layers.dense(step, action_dim, activation=tf.nn.relu)

			outputs = step
			selected_actions = tf.argmax(step, axis=1)

			dict_outputs['inputs'] = inputs
			dict_outputs['outputs'] = outputs
			dict_outputs['selected_actions'] = selected_actions

		return dict_outputs

	def build_copy_ops(self, pred_name, target_name):
		copy_ops = []

		pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=pred_name)
		target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_name)
		
		for pred_var, target_var in zip(pred_vars, target_vars):
			copy_ops.append(target_var.assign(pred_var.value()))

		return copy_ops

def calc_simple_reward(prev_x_pos, x_pos, done):

	if prev_x_pos < 0:
		return 0

	if int(x_pos / 20) > int(prev_x_pos / 20):
		print(x_pos, prev_x_pos)
		return 10

	return 0
		

def calc_reward(prev_info, info, done):
	if prev_info == None or info == None:
		return 0

	reward = 0

	if done:
		reward = info['x_pos']
		print(reward)

	return reward

	if done:
		if info['flag_get']:
			reward += 100
		else:
			reward -= 1000

	# reach to goal : True / False
	if prev_info['flag_get'] == False and info['flag_get'] == True:
		reward += 100

	# mario status : small, tall, fireball
	if prev_info['status'] != info['status']:
		status = ['small', 'tall', 'fireball']
		prev = status.index(prev_info['status'])
		cur = status.index(info['status'])
		assert(0 <= prev and prev < len(status) and 0 <= cur and cur < len(status))
		reward += (cur - prev) * 10

	# current stage : 1, 2, ..., 8
	info['stage']

	# distance from the start point : integer
	if prev_info['x_pos'] != info['x_pos']:
		prev = prev_info['x_pos']
		cur = info['x_pos']
		reward += (cur - prev) * 1

	# number of coins : integer
	if prev_info['coins'] != info['coins']:
		prev = prev_info['coins']
		cur = info['coins']
		reward += (cur - prev) * 5

	# how many lives : integer
#	info['life']
#	if prev_info['life'] != info['life']:
#		prev = prev_info['life']
#		cur = info['life']
#		reward += (cur - prev) * 50

	# score : integer
	info['score']

	# world : integer
	info['world']

	# remains time : integer
#	if prev_info['time'] != info['time']:
#		prev = prev_info['time']
#		cur = info['time']
#		reward += (cur - prev) * 5

	return reward

	
def train_dqn():
	# gym env
	env = gym_super_mario_bros.make('SuperMarioBros-v0')
	env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

	# generate args
	parser = argparse.ArgumentParser(description="SuperMarioBros")
	parser.add_argument('--replay_memory_total', default=100000, type=int, help="")
	parser.add_argument('--training_batch_size', default=64, type=int, help="")

	parser.add_argument('--action_dim', default=env.action_space.n, type=int, help="The number of available actions")

	parser.add_argument('--gamma', default=0.9, type=float, help="")
	parser.add_argument('--learning_rate', default=1e-3, type=float, help="")

	parser.add_argument('--epsilon', default=0.05, type=float, help="")

	args = parser.parse_args()

	# generate tf graph
	tf.reset_default_graph()

	dqn = DQN(args)
	replay_memory = ReplayMemory(args)

	# play!
	config = tf.ConfigProto()
	config.log_device_placement = False
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		saver = tf.train.Saver()

		tf.global_variables_initializer().run()
		done = True
		prev_info = None
		top_x_pos = -1
		prev_x_pos = 0
		prev_life = -1
		total_reward = 0
		total_steps = 0
		state = None
		next_state = None
		prev_top_record = -1
		for step in range(1000000):
			state = next_state

			if done:
				current_record = top_x_pos / total_steps if total_steps > 0 else 0.0
				if current_record > prev_top_record:
					print(current_record, prev_top_record)
					ckpt_path = saver.save(sess, "saved_top_dqn/train")
					print('top record! {} saved to {}'.format(prev_top_record, ckpt_path))
					prev_top_record = current_record

				print("done! at {}".format(top_x_pos))
				prev_info = None
				top_x_pos = -1
				prev_x_pos = 0
				prev_life = -1
				total_reward = 0
				total_steps = 0
				state = env.reset()


			# e-greedy
			if random.random() < args.epsilon:
				action = env.action_space.sample()
			else:
				action = dqn.next_action(sess, state)

			next_state, reward, done, info = env.step(action)
			if prev_life < 0:
				prev_life = info['life']

			x_pos = int(info['x_pos'] / 10)

			# re-calculate reward
			if info['flag_get']:
				print('FLAG!!! {}, {}'.format(x_pos, prev_x_pos))
				reward = 10.0
			else:
				reward = max((x_pos - prev_x_pos) * 5.0, -20.0)
			prev_x_pos = x_pos

			prev_info = info
			top_x_pos = max(top_x_pos, info['x_pos'])

			# total steps
			total_steps += 1

			replay_memory.push(state, action, reward, next_state, done)

			if len(replay_memory) < args.training_batch_size:
				continue
			
			loss = dqn.train(sess, *replay_memory.get_replays())
			if step % 10 == 0:
				dqn.update_target(sess)

			if step % 50 == 0:
				print('{} step : loss({:.4f}) x_pos({})'.format(step, loss, info['x_pos']))

			if step % 1000 == 0:
				ckpt_path = saver.save(sess, "saved_dqn/train", step)
				print('saved to {}'.format(ckpt_path))

	env.close()

def play_dqn():
	# gym env
	env = gym_super_mario_bros.make('SuperMarioBros-v0')
	env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

	# generate args
	parser = argparse.ArgumentParser(description="SuperMarioBros")
	parser.add_argument('--replay_memory_total', default=100000, type=int, help="")
	parser.add_argument('--training_batch_size', default=64, type=int, help="")

	parser.add_argument('--action_dim', default=env.action_space.n, type=int, help="The number of available actions")

	parser.add_argument('--gamma', default=0.99, type=float, help="")
	parser.add_argument('--learning_rate', default=1e-3, type=float, help="")

	parser.add_argument('--epsilon', default=0.05, type=float, help="")

	args = parser.parse_args()

	# generate tf graph
	tf.reset_default_graph()

	dqn = DQN(args)

	config = tf.ConfigProto()
	config.log_device_placement = False
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		saver = tf.train.Saver()

		tf.global_variables_initializer().run()
		tf.local_variables_initializer().run()
		done = True
		saver.restore(sess,tf.train.latest_checkpoint('./saved_top_dqn'))
		
		for step in range(1000000):
			if done:
				state = env.reset()

			action = dqn.next_action(sess, state)
			next_state, reward, done, info = env.step(action)
			if done:
				print(info['x_pos'], done)

#env.render()

def main():
	train_dqn()
#	play_dqn()

if __name__ == '__main__':
	main()

