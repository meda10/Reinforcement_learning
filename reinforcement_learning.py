import sys

import gym
import random
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from collections import deque


class Agent:
    def __init__(self, environment):
        self.is_discrete = type(environment.action_space) == gym.spaces.discrete.Discrete
        self.environment = environment

        if self.is_discrete:
            self.action_size = environment.action_space.n
            # print("Action size:", self.action_size)
        else:
            self.action_low = environment.action_space.low
            self.action_high = environment.action_space.high
            self.action_shape = environment.action_space.shape
            # print("Action range:", self.action_low, self.action_high)

    def get_action(self, state):
        if self.is_discrete:
            return random.choice(range(self.action_size))
        else:
            return np.random.uniform(self.action_low, self.action_high, self.action_shape)

    def train(self):
        pass


class QLearning(Agent):
    def __init__(self, environment, learning_rate=0.85, discount_rate=0.8, epsilon=1, decay_rate=0.005, num_episodes=10000):
        super().__init__(environment)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.decay_rate = decay_rate
        self.num_episodes = num_episodes
        self.q_table = np.zeros([environment.observation_space.n, environment.action_space.n])

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return super().get_action(state)
        else:
            return np.argmax(self.q_table[state, :])

    def train(self):
        for episode in range(self.num_episodes):
            state = self.environment.reset()
            done = False

            while not done:
                action = self.get_action(state)
                new_state, reward, done, info = self.environment.step(action)

                # Q-learning: Q(s,a) := Q(s,a) + learning_rate * (reward + discount_rate * max Q(s',a') - Q(s,a))
                update = reward + self.discount_rate * np.max(self.q_table[new_state, :]) - self.q_table[state, action]
                self.q_table[state, action] = self.q_table[state, action] + self.learning_rate * update
                if done:
                    self.epsilon = (1 - self.decay_rate) * self.epsilon

                state = new_state


class QLearningNuralNet(Agent):
    def __init__(self, environment, learning_rate=0.7, discount_rate=0.8, epsilon=1, min_epsilon=0.1, decay_rate=0.005, num_episodes=10000):
        super().__init__(environment)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.decay_rate = decay_rate
        self.num_episodes = num_episodes

        self.model = self.build_model()
        self.replay_memory = deque(maxlen=50_000)
        self.min_replay_size = 1000

    def build_model(self):
        learning_rate = 0.001
        init = keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_shape=[1], activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(self.environment.action_space.n, activation='linear', kernel_initializer=init))
        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])
        model.build()
        return model

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return super().get_action(state)
        else:
            encoded = state
            encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
            predicted = self.model.predict(encoded_reshaped).flatten()
            return np.argmax(predicted)

    def train(self):
        steps_to_update_target_model = 0
        target_model = self.build_model()
        target_model.set_weights(self.model.get_weights())

        for episode in range(self.num_episodes):
            total_training_rewards = 0
            state = self.environment.reset()
            done = False

            while not done:
                steps_to_update_target_model += 1

                action = self.get_action(state)
                new_state, reward, done, info = self.environment.step(action)

                self.replay_memory.append([state, action, reward, new_state, done])

                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 4 == 0 or done:
                    self.train_nural_net(target_model)

                state = new_state
                total_training_rewards += reward

                if done:
                    print('Total training rewards: {} after n steps = {} with final reward = {}'.format(
                        total_training_rewards, episode, reward))
                    total_training_rewards += 1

                    if steps_to_update_target_model >= 100:
                        print('Copying main network weights to the target network weights')
                        target_model.set_weights(self.model.get_weights())
                        steps_to_update_target_model = 0
                    break

            self.epsilon = self.min_epsilon + (self.epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

    def train_nural_net(self, target_model):
        if len(self.replay_memory) < self.min_replay_size:
            return

        batch_size = 64 * 2
        mini_batch = random.sample(self.replay_memory, batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (state, action, reward, new_state, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + self.discount_rate * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - self.learning_rate) * current_qs[action] + self.learning_rate * max_future_q

            X.append(state)
            Y.append(current_qs)
        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

    def __del__(self):
        pass


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def run(env, agent):
    num_episodes = 100
    total_reward = 0
    total_epochs = 0
    total_penalties = 0

    for episode in range(num_episodes):
        state = env.reset()
        epochs = 0
        done = False

        while not done:
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)

            total_reward += reward
            epochs += 1

            if reward == -10:
                total_penalties += 1

            env.render()
            # print("Step: {} | score: {}".format(epochs, total_reward))

            if done:
                break

        total_epochs += epochs

    env.close()
    print("Results after {} episodes".format(num_episodes))
    print("Reward: {}".format(total_reward))
    print("Average time-steps per episode: {}".format(total_epochs / num_episodes))
    print("Average penalties per episode: {}".format(total_penalties / num_episodes))


if __name__ == "__main__":
    # env = gym.make("MountainCar-v0")
    env = gym.make("Taxi-v3")
    # print(env.observation_space)
    # print(env.action_space.n)
    # sys.exit(1)
    agent = QLearningNuralNet(env)
    # agent = QLearning(env)
    # agent = Agent(env)
    agent.train()
    run(env, agent)

    # print("The observation space: {}".format(env.observation_space))
    # print("The action space: {}".format(env.action_space))
    # input("Press Enter to watch trained agent...")
