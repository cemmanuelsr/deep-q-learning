import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import gym
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from collections import deque
from Model import Model
from DeepQLearning import DeepQLearning
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(prog='Breakout')
parser.add_argument('-t', '--train', action='store_true')
parser.add_argument('-m', '--model', type=str, default='breakout')
args = parser.parse_args()

env = gym.make('ALE/Breakout-v5')
np.random.seed(42)

print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

if args.train:
    model = Model(env)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')

    gamma = 0.99 
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_dec = 0.99
    episodes = 500
    batch_size = 64
    memory = deque(maxlen=500000) 

    algorithm = DeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model)
    print('Training with DQN approach')
    rewards = algorithm.train()

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('# Rewards')
    plt.title('# Rewards vs Episodes')
    plt.savefig(f"../results/{args.model}.jpg")     
    plt.close()

    model.save(f'../models/{args.model}')
else:
    state = env.reset()
    model = keras.models.load_model(f'../models/{args.model}', compile=False)
    done = False
    rewards = 0
    steps = 0

    while not done and steps < 500:
        Q_values = model.predict(state[np.newaxis], verbose=0)
        action = np.argmax(Q_values[0])
        state, reward, done, info = env.step(action)
        rewards += reward
        env.render()
        steps += 1