import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.keras import layers


class D3QN(keras.Model):
    def __init__(self, action_space):
        super(D3QN, self).__init__()
        self.dense1 = layers.Dense(units=32, activation='relu')
        self.dense2 = layers.Dense(units=64, activation='relu')
        self.V = layers.Dense(units=1, activation="linear")
        self.A = layers.Dense(units=action_space, activation="linear")

    
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        # Q = Value +( Advantage - Average Advantage)
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    
    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A