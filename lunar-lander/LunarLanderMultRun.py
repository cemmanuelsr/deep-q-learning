import gym
import numpy as np
from tensorflow import keras
import argparse

parser = argparse.ArgumentParser(prog='LunarLanderMultRun')
parser.add_argument('-n', '--number', type=int, default=100)
parser.add_argument('-m', '--model', type=str, default='lunar_lander_deep_qlearning')
args, _ = parser.parse_known_args()

env = gym.make('LunarLander-v2').env
model = keras.models.load_model(f'../models/{args.model}', compile=False)

rewards = 0
finished = 0

for i in range(args.number):    
    state = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        Q_values = model.predict(state[np.newaxis], verbose=0)
        action = np.argmax(Q_values[0])
        state, reward, done, info = env.step(action)
        steps += 1

    if done:
        finished += 1

    rewards += reward

print(f'Playing {args.number} different states, finished {finished} of them and get {rewards} of rewards')
