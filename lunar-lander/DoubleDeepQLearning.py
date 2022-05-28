import numpy as np
import random
from keras.activations import relu, linear
from tensorflow.keras.models import clone_model

class DoubleDeepQLearning:

    def __init__(self, env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, max_steps, model, update_target_frequency):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory = memory
        self.max_steps = max_steps
        self.q_network = model
        self.target = clone_model(model)
        self.target.compile(loss='mse', optimizer='adam')
        self.update_target_frequency = update_target_frequency

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        action = self.q_network.predict(state, verbose=0)
        return np.argmax(action[0])

    # cria uma memoria longa de experiencias
    def experience(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal)) 

    def experience_replay(self):
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size) #escolha aleatoria dos exemplos
            states = np.array([i[0] for i in batch])
            actions = np.array([i[1] for i in batch])
            rewards = np.array([i[2] for i in batch])
            next_states = np.array([i[3] for i in batch])
            terminals = np.array([i[4] for i in batch])

            # np.squeeze(): Remove single-dimensional entries from the shape of an array.
            # Para se adequar ao input
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)

            # usando o modelo para selecionar as melhores acoes
            next_max = np.amax(self.target.predict_on_batch(next_states), axis=1)
            
            targets = rewards + self.gamma * (next_max) * (1 - terminals)
            targets_full = self.target.predict_on_batch(states)
            indexes = np.array([i for i in range(self.batch_size)])
            
            # usando os q-valores para atualizar os pesos da rede
            targets_full[[indexes], [actions]] = targets
            self.q_network.fit(states, targets_full, epochs=1, verbose=0)
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_dec

    def train(self):
        rewards = []
        for i in range(self.episodes+1):
            state = self.env.reset()
            state = np.reshape(state, (1, self.env.observation_space.shape[0]))
            score = 0
            for _ in range(self.max_steps):
                action = self.select_action(state)
                self.env.render()
                next_state, reward, terminal, _ = self.env.step(action)
                score += reward
                next_state = np.reshape(next_state, (1, self.env.observation_space.shape[0]))
                self.experience(state, action, reward, next_state, terminal)
                state = next_state
                self.experience_replay()
                if terminal:
                    print(f'Episódio: {i+1}/{self.episodes}. Score: {score}')
                    break
            rewards.append(score)
            if i % self.update_target_frequency == 0:
                self.target.set_weights(self.q_network.get_weights())

        return rewards
