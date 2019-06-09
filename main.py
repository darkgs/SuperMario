
import os

import random
from optparse import OptionParser

import tensorflow as tf
import numpy as np

from nes_py.wrappers import JoypadSpace

import gym_super_mario_bros

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from DQN import DQN
from SplitDQN import SplitDQN
from SADQN import SplitAttnDQN

from ReplayMemory import ReplayMemory

parser = OptionParser()
parser.add_option('-a', '--replay_memory_total', dest='replay_memory_total',  default=50000, type=int)
parser.add_option('-t', '--training_batch_size', dest='training_batch_size', default=16, type=int)

parser.add_option('-g', '--gamma', dest='gamma', default=0.9, type=float)
parser.add_option('-l', '--learning_rate', dest='learning_rate', default=1e-3, type=float)

parser.add_option('-e', '--epsilon', dest='epsilon', default=0.02, type=float)

parser.add_option('-s', '--max_steps', dest='max_steps', default=1000000, type=int)

parser.add_option('-m', '--model_name', dest='model_name', default='DQN', type=str)
parser.add_option('-p', '--play_mode', dest='play_mode', action="store_true")
parser.add_option('-r', '--reward_mode', dest='reward_mode', default='R0', type=str)

def write_log(log_file, log):

	dir_path = os.path.dirname(os.path.abspath(log_file))

	if not os.path.exists(dir_path):
		os.system('mkdir -p {}'.format(dir_path))

	with open(log_file, 'a') as f_log:
		f_log.write('{}\n'.format(log))


class Mario(object):

	def __init__(self, args, env):
		self._args = args
		self._env = env


	def get_rewards(self, prev_info, info):
		"""
		info['flag_get'] : Boolean
		info['coins'] : Integer
		info['x_pos'] : Integer
		info['status'] : String in ['small', 'tall', 'fireball']
		info['stage'] : Integer
		info['score'] : Integer
		info['world'] : Integer
		info['life'] : Integer
		info['time'] : Integer
		"""
		# re-calculate reward
		reward = 0.0
		if self._args.reward_mode == 'R0':
			reward = 0.0
		elif self._args.reward_mode == 'R1':
			reward = -1.
		elif self._args.reward_mode == 'R2':
			reward = -1.

		if prev_info == None or info == None:
			return reward

		if self._args.reward_mode == 'R0':
			x_pos = int(info['x_pos'] / 10)
			prev_x_pos = int(prev_info['x_pos'] / 10)
		elif self._args.reward_mode == 'R1':
			x_pos = int(info['x_pos'] / 3)
			prev_x_pos = int(prev_info['x_pos'] / 3)
		elif self._args.reward_mode == 'R2':
			x_pos = int(info['x_pos'])
			prev_x_pos = int(prev_info['x_pos'])

		c_time = int(info['time'])
		prev_time = int(prev_info['time'])

		if info['flag_get']:
			reward += 50.0
		else:
			reward += max((x_pos - prev_x_pos) * 5.0, -20.0)

		if prev_time < 0:
			prev_time = c_time

		if prev_time > c_time:
			reward += -1.0

		return reward

	def train(self, model_name):
		# generate tf graph
		tf.reset_default_graph()

		assert(model_name in ['DQN', 'DQN0', 'DQN1', 'DQN2', 'DQN3',
				'SDQN', 'SDQN0', 'SDQN1', 'SDQN2', 'SDQN3',
				'SADQN', 'SADQN0', 'SADQN1', 'SADQN2', 'SADQN3'])

		if model_name.startswith('DQN'):
			model = DQN(self._args, model_name)
		elif model_name.startswith('SDQN'):
			model = SplitDQN(self._args, model_name)
		elif model_name.startswith('SADQN'):
			model = SplitAttnDQN(self._args, model_name)
		else:
			assert(False)
		reward_mode = self._args.reward_mode

		replay_memory = ReplayMemory(self._args)

		config = tf.ConfigProto()
		config.log_device_placement = False
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			saver = tf.train.Saver()

			tf.global_variables_initializer().run()
			tf.local_variables_initializer().run()

			done = True
			prev_info = None
			top_x_pos = -1
			prev_test_step = -1

			for step in range(self._args.max_steps):
				if done:
					prev_info = None
					state = self._env.reset().copy()

				# e-greedy
#	if random.random() < self._args.epsilon:
				if random.random() < (self._args.epsilon * np.exp(-1./10000.*float(step)) + 0.005):
					action = self._env.action_space.sample()
				else:
					action = model.next_action(sess, state)

				# step!
				next_state, reward, done, info = self._env.step(action)

				reward = self.get_rewards(prev_info, info)

				# train with ReplayMemory
				replay_memory.push(state, action, reward, next_state, done)

				if len(replay_memory) < self._args.training_batch_size:
					continue

				loss = model.train(sess, *replay_memory.get_replays())
				if step % 50 == 0:
					model.update_target(sess)

				if step % 50 == 0:
					print('{} step : loss({:.4f}) x_pos({})'.format(step, loss, info['x_pos']))
					write_log('saved_{}_{}/loss.txt'.format(model_name, reward_mode), '{},{}'.format(step,loss))

				if done and ((prev_test_step < 0) or (step > prev_test_step + 1000)):
					test_x_pos, reward_sum = self.test(sess, model)
					done = True
					prev_test_step = step
					write_log('saved_{}_{}/reward.txt'.format(model_name, reward_mode), '{},{}'.format(step,reward_sum))
					if test_x_pos > top_x_pos:
						top_x_pos = test_x_pos
						ckpt_path = saver.save(sess, 'saved_{}_{}/top_{}/model'.format(model_name, reward_mode,  top_x_pos))
						print('new record! top_x_pos({}), reward_sum({}), saved : {}'.format(top_x_pos, reward_sum, ckpt_path))
					else:
						print('what a shame ! test x_pos is {}'.format(test_x_pos))

				# next action
				state = next_state.copy()
				prev_info = info.copy()

	def test(self, sess, model):
		done = False

		top_x_pos = -1
		prev_info = None
		reward_sum = 0.0

		state = self._env.reset().copy()
		while(not done):
			action = model.next_action(sess, state)
			next_state, reward, done, info = self._env.step(action)

			if done:
				break

			if prev_info != None and abs(prev_info['x_pos']-info['x_pos']) < 20:
				top_x_pos = max(top_x_pos, info['x_pos'])
				reward = self.get_rewards(prev_info, info)
				reward_sum += reward

			# next action
			state = next_state.copy()
			prev_info = info.copy()

		return top_x_pos, reward_sum

	def play(self, model_name):
		# generate tf graph
		tf.reset_default_graph()

		assert(model_name in ['DQN', 'DQN0', 'DQN1', 'DQN2', 'DQN3'])

		if model_name.startswith('DQN'):
			model = DQN(self._args, model_name)
		else:
			assert(False)

		replay_memory = ReplayMemory(self._args)

		config = tf.ConfigProto()
		config.log_device_placement = False
		config.gpu_options.allow_growth = True

		with tf.Session(config=config) as sess:
			saver = tf.train.Saver()
			saver.restore(sess,tf.train.latest_checkpoint('./saved_DQN1/top_900'))

			done = True

			while(True):
				if done:
					state = self._env.reset().copy()
				action = model.next_action(sess, state)
				next_state, reward, done, info = self._env.step(action)

				# next action
				state = next_state.copy()

				self._env.render()

def main():
	# gym env
	env = gym_super_mario_bros.make('SuperMarioBros-v0')
	env = JoypadSpace(env, SIMPLE_MOVEMENT)

	# generate args
	options, args = parser.parse_args()
	setattr(options, 'action_dim', env.action_space.n)

	model_name = options.model_name
	play_mode = options.play_mode

	mario = Mario(options, env)
	if play_mode:
		mario.play(model_name)
	else:
		mario.train(model_name)

	# close env
	env.close()

if __name__ == '__main__':
	main()

