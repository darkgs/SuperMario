
import numpy as np

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

