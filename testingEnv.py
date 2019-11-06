import gym
env = gym.make('MountainCar-v0')
env.reset()
# for _ in range(1000):
#     env.render()
#     obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
#     print(_, obs, reward, done)

done = False
while not done:
	obs, reward, done, info = env.step(env.action_space.sample())
	print(obs, reward, done)
	env.render()



env.close()