import gym
import numpy as np

env = gym.make('MountainCar-v0')

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 5000

### create discrete table for states and actions
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decaying_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

### function to discretize continuous states from env
def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

### MAIN LOOP
for episode in range(EPISODES):
	if episode % SHOW_EVERY == 0:
		render = True
		print(episode)
	else:
		render = False
	
	discrete_state = get_discrete_state(env.reset())
	done = False
	while not done:

		### AGENT
		### explore/exploit
		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)

		### RESPONSE FROM ENV
		new_state, reward, done, _ = env.step(action)
		new_discrete_state = get_discrete_state(new_state)

		if render:
			env.render()

		### AGENT LEARNS
		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action,)]
			####here is the learning:
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + (DISCOUNT * max_future_q))
			q_table[discrete_state + (action,)] = new_q
		elif new_state[0] >= env.goal_position:
			print("We made it in episode: ", episode)			
			q_table[discrete_state + (action,)] = 0
		discrete_state = new_discrete_state

		### epsilon decay is based on episode number
		if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
			epsilon -= epsilon_decaying_value

env.close()