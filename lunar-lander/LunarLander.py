import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from collections import deque
from Model import Model
from tensorflow.keras.optimizers import Adam
from DeepQLearning import DeepQLearning
import argparse

parser = argparse.ArgumentParser(prog='LunarLander')
parser.add_argument('-t', '--train', type=bool, default=True)
parser.add_argument('-r', '--result', type=str, default='lunar_land')
parser.add_argument('-m', '--model', type=str, default='lunar_lander_deep_qlearning.jpg')
args = parser.parse_args()

env = gym.make('LunarLander-v2')
np.random.seed(42)

print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

if args.train:
    model = Model(env)
    model.summary()
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

    gamma = 0.99 
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_dec = 0.99
    episodes = 1000
    batch_size = 64
    memory = deque(maxlen=500000) 

    DQN = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model)
    rewards = DQN.train()

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('# Rewards')
    plt.title('# Rewards vs Episodes')
    plt.savefig(f"../results/{args.result}")     
    plt.close()

    model.save(f'../models/{args.model}')
else:
    state = env.reset()
    model = keras.models.load_model(f'../models/{args.model}', compile=False)
    done = False
    rewards = 0
    steps = 0

    while not done and steps < 250:
        Q_values = model.predict(state[np.newaxis])
        action = np.argmax(Q_values[0])
        state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        steps += 1
