
import tensorflow as tf
import numpy as np

class SplitDQN(object):
	def __init__(self, args, model_name):
		self._args = args
		self._model_name = model_name

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

			step_down = step[:,:,192:,:]
			step_center = step[:,46:110,:,:]

			# conv layers
#			if self._model_name == 'DQN0':
#			elif self._model_name == 'DQN1':
#			elif self._model_name == 'DQN2':
#			elif self._model_name == 'DQN3':
			# conv layers for all
			step = conv_layer(inputs=step,
					conv_filters=16, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 240, 256, x)

			step = conv_layer(inputs=step,
					conv_filters=24, conv_kernel_size=3, conv_strides=(2,2), conv_padding='same',
					pool_size=(2,2), pool_strides=(1,1), pool_padding='same')	# (?, 120, 128, x)

			step = conv_layer(inputs=step,
					conv_filters=24, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 60, 64, x)

			step = conv_layer(inputs=step,
					conv_filters=32, conv_kernel_size=5, conv_strides=(2,2), conv_padding='same',
					pool_size=(4,4), pool_strides=(1,1), pool_padding='same')	# (?, 30, 32, x)

			step = conv_layer(inputs=step,
					conv_filters=32, conv_kernel_size=5, conv_strides=(1,1), conv_padding='same',
					pool_size=(4,4), pool_strides=(2,2), pool_padding='same')	# (?, 15, 16, x)

			step = conv_layer(inputs=step,
					conv_filters=48, conv_kernel_size=3, conv_strides=(1, 1), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 8, 8, x)

			if self._model_name == 'SDQN0':
				# conv layers for down
				step_down = conv_layer(inputs=step_down,
					conv_filters=16, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 240, 64, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=24, conv_kernel_size=3, conv_strides=(2,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 120, 64, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=24, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,1), pool_padding='same')	#(?, 60, 64, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=32, conv_kernel_size=3, conv_strides=(2,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 30, 32, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=32, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,2), pool_padding='same')	#(?, 15, 16, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=48, conv_kernel_size=3, conv_strides=(1, 1), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 8, 8, x)

				# conv layers for center
				step_center = conv_layer(inputs=step_center,
					conv_filters=16, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 64, 256, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=24, conv_kernel_size=3, conv_strides=(1,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 64, 128, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=24, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,2), pool_padding='same')	#(?, 64, 64, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=32, conv_kernel_size=3, conv_strides=(2,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 32, 32, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=32, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,2), pool_padding='same')	#(?, 16, 16, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=48, conv_kernel_size=3, conv_strides=(1, 1), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 8, 8, x)

				step = tf.reshape(step, [-1, 8*8, 48])
				step_down = tf.reshape(step_down, [-1, 8*8, 48])
				step_center = tf.reshape(step_center, [-1, 8*8, 48])

				step = tf.concat([step, step_down, step_center], axis=1)
				step = tf.reshape(step, [-1, 192*48])

			elif self._model_name == 'SDQN1':
				# conv layers for down
				step_down = conv_layer(inputs=step_down,
					conv_filters=8, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 240, 64, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=12, conv_kernel_size=3, conv_strides=(2,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 120, 64, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=12, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,1), pool_padding='same')	#(?, 60, 64, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=16, conv_kernel_size=3, conv_strides=(2,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 30, 32, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=16, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,2), pool_padding='same')	#(?, 15, 16, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=24, conv_kernel_size=3, conv_strides=(1, 1), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 8, 8, x)

				# conv layers for center
				step_center = conv_layer(inputs=step_center,
					conv_filters=8, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 64, 256, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=12, conv_kernel_size=3, conv_strides=(1,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 64, 128, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=12, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,2), pool_padding='same')	#(?, 64, 64, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=16, conv_kernel_size=3, conv_strides=(2,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 32, 32, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=16, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,2), pool_padding='same')	#(?, 16, 16, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=24, conv_kernel_size=3, conv_strides=(1, 1), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 8, 8, x)

				step = tf.reshape(step, [-1, 8*8, 48])
				step_down = tf.reshape(step_down, [-1, 8*8, 24])
				step_center = tf.reshape(step_center, [-1, 8*8, 24])

				step_side = tf.concat([step_down, step_center], axis=2)
				step = tf.concat([step, step_side], axis=1)
				step = tf.reshape(step, [-1, 128*48])

			elif self._model_name == 'SDQN2':
				# conv layers for down
				step_down = conv_layer(inputs=step_down,
					conv_filters=8, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 240, 64, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=12, conv_kernel_size=3, conv_strides=(2,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,1), pool_padding='same')	#(?, 60, 64, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=16, conv_kernel_size=3, conv_strides=(2,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,2), pool_padding='same')	#(?, 15, 16, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=24, conv_kernel_size=3, conv_strides=(1, 1), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 8, 8, x)

				# conv layers for center
				step_center = conv_layer(inputs=step_center,
					conv_filters=8, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 64, 256, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=12, conv_kernel_size=3, conv_strides=(1,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,2), pool_padding='same')	#(?, 64, 64, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=16, conv_kernel_size=3, conv_strides=(2,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,2), pool_padding='same')	#(?, 16, 16, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=24, conv_kernel_size=3, conv_strides=(1, 1), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 8, 8, x)

				step = tf.reshape(step, [-1, 8*8, 48])
				step_down = tf.reshape(step_down, [-1, 8*8, 24])
				step_center = tf.reshape(step_center, [-1, 8*8, 24])

				step_side = tf.concat([step_down, step_center], axis=2)
				step = tf.concat([step, step_side], axis=1)
				step = tf.reshape(step, [-1, 128*48])

			if self._model_name == 'SDQN3':
				# conv layers for down
				step_down = conv_layer(inputs=step_down,
					conv_filters=16, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 240, 64, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=24, conv_kernel_size=3, conv_strides=(2,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,1), pool_padding='same')	#(?, 60, 64, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=32, conv_kernel_size=3, conv_strides=(2,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,2), pool_padding='same')	#(?, 15, 16, x)

				step_down = conv_layer(inputs=step_down,
					conv_filters=48, conv_kernel_size=3, conv_strides=(1, 1), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 8, 8, x)

				# conv layers for center
				step_center = conv_layer(inputs=step_center,
					conv_filters=16, conv_kernel_size=3, conv_strides=(1,1), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,1), pool_padding='same')	#(?, 64, 256, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=24, conv_kernel_size=3, conv_strides=(1,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(1,2), pool_padding='same')	#(?, 64, 64, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=32, conv_kernel_size=3, conv_strides=(2,2), conv_padding='same',
					pool_size=(3,3), pool_strides=(2,2), pool_padding='same')	#(?, 16, 16, x)

				step_center = conv_layer(inputs=step_center,
					conv_filters=48, conv_kernel_size=3, conv_strides=(1, 1), conv_padding='same',
					pool_size=(2,2), pool_strides=(2,2), pool_padding='same')	# (?, 8, 8, x)

				step = tf.reshape(step, [-1, 8*8, 48])
				step_down = tf.reshape(step_down, [-1, 8*8, 48])
				step_center = tf.reshape(step_center, [-1, 8*8, 48])

				step = tf.concat([step, step_down, step_center], axis=1)
				step = tf.reshape(step, [-1, 192*48])

			# MLP
			step = tf.layers.dense(step, 128, activation=tf.nn.relu)
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

