from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam

def Model(env):
    model = Sequential()
    model.add(Dense(512, activation=relu, input_dim=env.observation_space.shape[0]))
    model.add(Dense(256, activation=relu))
    model.add(Dense(env.action_space.n, activation=linear))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

