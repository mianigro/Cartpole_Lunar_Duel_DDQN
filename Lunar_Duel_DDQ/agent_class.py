# Python imports
import os
import shutil
import random
from csv import writer

# Third party imports
import numpy as np
import tensorflow as tf

# Module imports
from replay_buffer import Replay_Buffer
from tf_model_dense import D3QN


class Game_Agent:
    def __init__(self, environment):
        # Environment setup from the input environment
        self.env_class = environment
        self.env = environment.env
        self.state_size = environment.state_size
        self.action_size = environment.action_space  
        self.view_render = False

        # DDQN hyperparameters
        self.gamma = 0.99 
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.target_update_freq = 500
        self.it_count = 0

        # Training hyperparameters
        self.EPISODES = 5000
        self.batch_size = 64
        self.e_buffer = 0
        self.episode_count = 0
        self.train_start_counter = 0
        self.learning_rate = 0.001
        self.training_accuracy = []
        
        # Model saving
        self.save_model_status = False
        self.load_model_status = False
        self.episodes_to_save = 10
        self.best_score_saved = -9999999999

        # Replay memory hyperparameters
        self.mem_size = 50000
        self.memory = Replay_Buffer(
            self.state_size, 
            self.mem_size, 
            self.batch_size)
        self.train_start = self.mem_size

        # Build the models
        self.q_eval = D3QN(self.action_size)
        self.q_eval.compile(
            loss='mse', 
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate),
            metrics=["mean_absolute_error"])
        self.q_next = D3QN(self.action_size)
        self.q_next.compile(loss='mse',
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate))

        # Set main and target networks to same weight to commence
        self.update_target()


    def to_memory(self, state, action, reward, next_state, done):
        self.memory.store_memory(state, action, reward, next_state, done)


    # To easily determine if explor or exploit
    def do_action(self, state):
        if np.random.random() <= self.epsilon:
            action = random.randint(0, self.action_size-1)
        else:
            state = np.reshape(state, (1, self.state_size))
            
            # Dueling returns advantage of action to pick the action
            action = np.argmax(self.q_eval.advantage(state))

        return action


    def replay_train(self):
        state, action, reward, next_state, done = self.memory.sample_buffer()

        # Get Q value of current state with eval network for each action
        q_pred = np.array(self.q_eval(state))
        # Need value of next state with target netwrok for each action
        q_next = np.array(self.q_next(next_state))
        # Get Q value of next state on eval for each action
        #   This is the max action on each state that could be taken
        q_eval = np.array(self.q_eval(next_state))

        for i in range(self.batch_size):
            if done[i]:
                q_pred[i][int(action[i])] = reward[i]
            else:
                # Duelling version of Bellman equation 
                q_pred[i][int(action[i])] = \
                    reward[i] + self.gamma * q_next[i][np.argmax(q_eval[i])]
        
        history = self.q_eval.fit(state, q_pred, epochs=1, verbose=0)
        self.training_accuracy.append(history.history["mean_absolute_error"][0])

        # Updates after specific amount of training itterations
        #   to keep consistnetly with target weights
        #   decays epsilon at the same time
        self.it_count += 1               
        if self.it_count == self.target_update_freq:
            self.update_target()
            self.decay_epsilon()
            self.it_count = 0


    def update_target(self):
        if len(self.training_accuracy) > 0:
            accuracy_history = \
                sum(self.training_accuracy)/len(self.training_accuracy)
            self.training_accuracy = []
            print("MAE: ", accuracy_history)  
               
        self.q_next.set_weights(self.q_eval.get_weights())


    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def save(self):
        # Don't want to keep older models
        if os.path.isdir("model"):
            print("Deleting model")
            shutil.rmtree('model')
        
        if os.path.isdir("model_target"):
            print("Deleting target model")
            shutil.rmtree('model_target')

        print("Saving models")
        self.q_eval.save_weights("model/model")
        self.q_next.save_weights("model_target/model_target")


    def load_model(self):
        if os.path.isdir("model") and os.path.isdir("model_target"):
            self.q_eval.load_weights("model/model")
            self.q_next.load_weights("model_target/model_target")
            print("Previouse weights loaded.")
        
        else:
            print("No model to load")


    def log_info(self, episode, epsilon, score):
        new_row = [episode, epsilon, score]

        with open('log_file.csv','a') as log_file:
            writer_object = writer(log_file)
            writer_object.writerow(new_row)