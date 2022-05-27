import tensorflow as tf
import keras

def Model(env):
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu',use_bias=False, input_shape=env.observation_space.shape),
        keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu',use_bias=False),
        keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu',use_bias=False),
        keras.layers.Flatten(),
        keras.layers.Dense(env.action_space.n, activation='softmax')
    ])

    return model
