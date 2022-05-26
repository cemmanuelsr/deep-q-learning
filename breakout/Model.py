import keras

def Model(env):
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", input_shape=env.observation_space.shape),
        keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(env.action_space.n, activation="softmax")
    ])

    return model
