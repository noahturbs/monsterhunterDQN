import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
from ahk.window import Window

import struct
import numpy as np
class DQN:
    def __init__(self):
        self.memory  = deque(maxlen=500)

        self.gamma = 0.993           #controls reward
        self.epsilon = 1.0          #controls randomness
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.learning_rate = 0.01   #large is good at beginning/generalization. small is good to optimize but bad at finding a solution
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
            model   = Sequential()
            #state_shape  = 23 # however many memory locations we are watching.
            model.add(Dense(22, input_shape=(22,) , activation="relu"))
            #model.add(Dense(46, activation="relu"))
            model.add(Dense(44, activation="relu"))
            model.add(Dense(44, activation="relu"))

            model.add(Dense(38))# however many possible moves we discretized. 54. final
            model.compile(loss="mean_squared_error",
                optimizer=Adam(lr=self.learning_rate))
            return model

    def act(self, state):
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            #print(np.argmax(self.model.predict(state)[0]))
            if np.random.random() < self.epsilon:
                return np.random.randint(0,37)                            #return self.env.action_space.sample()
            return np.argmax(self.model.predict(state)[0])


    def remember(self, state, action, reward, new_state, done):
            self.memory.append([state, action, reward, new_state, done])

    def replay(self):
            batch_size = 32
            if len(self.memory) < batch_size:
                return

            samples = random.sample(self.memory, batch_size)
            for sample in samples:
                state, action, reward, new_state, done = sample
                target = self.target_model.predict(state)
                if done:
                    target[0][action] = reward
                else:
                    Q_future = max(self.target_model.predict(new_state)[0])
                    target[0][action] = reward + Q_future * self.gamma
                self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)
    def set_epsilon(self, value):
        self.epsilon = value
