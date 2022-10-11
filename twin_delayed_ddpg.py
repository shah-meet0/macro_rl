import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_capacity=100000, batch_size=256):
        self.old_state = np.zeros((buffer_capacity, state_size))
        self.action_taken = np.zeros((buffer_capacity, action_size))
        self.reward_acquired = np.zeros((buffer_capacity, 1))
        self.next_state = np.zeros((buffer_capacity, state_size))
        self.buffer_capacity = buffer_capacity
        self.buffer_count = 0
        self.batch_size = batch_size

    def record(self, state, action, reward, next_state_exp):
        index = self.buffer_count % self.buffer_capacity
        self.old_state[index] = tf.cast(state, tf.float32) # State must be an array of type [[2,3,4]] for example.
        self.action_taken[index] = tf.cast(action, tf.float32)
        self.reward_acquired[index] = tf.cast(reward, tf.float32)
        self.next_state[index] = tf.cast(next_state_exp, tf.float32)
        self.buffer_count += 1

    def get_batch(self):
        record_range = min(self.buffer_count, self.buffer_capacity)
        assert record_range > self.batch_size
        indices = np.random.choice(record_range, self.batch_size)
        old_state_batch = tf.convert_to_tensor(self.old_state[indices], tf.float32)
        action_batch = tf.convert_to_tensor(self.action_taken[indices], tf.float32)
        reward_batch = tf.convert_to_tensor(self.reward_acquired[indices], tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state[indices], tf.float32)
        return old_state_batch, action_batch, reward_batch, next_state_batch

    def __len__(self):
        return min(self.buffer_count, self.buffer_capacity)

class GaussianNoise:
    pass

class OUNoise:
    pass

class TD3:
    # Maybe change how noise works
    def __init__(self, env, actor, critic, replay_buffer, periods=10000,
                 gamma=0.99, actor_lr= 0.001, critic_lr= 0.001, polyak=0.995):
        self.env = env
        self.gamma = gamma
        self.periods= periods
        self.batch_size = replay_buffer.batch_size

        self.actor = actor
        self.target_actor = tf.keras.models.clone_model(actor)
        self.target_actor.set_weights(self.actor.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.critic_1 = critic
        self.critic_2 = tf.keras.models.clone_model(critic)

        self.target_critic_1 = tf.keras.models.clone_model(self.critic_1)
        self.target_critic_2 = tf.keras.models.clone_model(self.critic_2)

        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.critic_optimizer_1 = tf.keras.optimizers.Adam(critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(critic_lr)

        self.polyak = polyak
        self.replay_buffer = replay_buffer


    def train_actor(self, old_state_batch):
        with tf.GradientTape() as tape:
            actions = self.actor(old_state_batch, training=True)
            q = self.critic_1([old_state_batch, actions], training=True)
            actor_loss = -1 * tf.reduce_mean(q)
        grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grad, self.actor.trainable_variables))

    def train_critics(self, old_state_batch, action_batch, reward_batch, next_state_batch):
        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(next_state_batch)
            noise = np.clip(np.random.normal(0, 2, size=target_actions.shape), -2, 2)
            target_actions = tf.clip_by_value(target_actions + noise, 0, 40) # Should be adjustable
            target_q_1 = self.target_critic_1([next_state_batch, target_actions])
            target_q_2 = self.target_critic_2([next_state_batch, target_actions])
            y  = reward_batch + self.gamma * tf.minimum(target_q_1, target_q_2)
            pred_1 = self.critic_1([old_state_batch, action_batch])
            pred_2 = self.critic_2([old_state_batch, action_batch])
            loss_1 = tf.reduce_mean(tf.square(y-pred_1))
            loss_2 = tf.reduce_mean(tf.square(y-pred_2))

        grad1 = tape.gradient(loss_1, self.critic_1.trainable_variables)
        grad2 = tape.gradient(loss_2, self.critic_2.trainable_variables)

        del tape
        self.critic_optimizer_1.apply_gradients(zip(grad1, self.critic_1.trainable_variables))
        self.critic_optimizer_2.apply_gradients(zip(grad2, self.critic_2.trainable_variables))

    def execute(self):
        obs, info = self.env.reset()
        state = np.array([obs])
        p = 50
        J = 6
        d = 2
        done = False
        iteration = 0
        for t in range(self.periods):
            if done:
                iteration += 1
                obs, info = self.env.reset()
                state = np.array([obs])
                final_cap = info['History']['Capital'][self.env.time_periods -2]
                print(f'Episode {iteration} reward: {info["Cumulative Reward"]}, final_cap: {final_cap}')

            if t < 10000:
                act = self.env.action_space.sample()
            else:
                act = self.policy(state)[0]

            obs, r, done, x, info = self.env.step(act)
            new_state = np.array([obs])
            final_act = np.array([act])
            r = np.array([[r]])
            self.replay_buffer.record(state, final_act, r, new_state)
            state = new_state

            if t > self.batch_size * 5:
                if t % p == 0:
                    for j in range(J):
                        old_states, actions, rewards, new_states =self.replay_buffer.get_batch()
                        self.train_critics(old_states, actions, rewards, new_states)

                        if j % d == 0:
                            self.train_actor(old_states)
                            self.update_target(self.actor, self.target_actor)
                            self.update_target(self.critic_1, self.target_critic_1)
                            self.update_target(self.critic_2, self.target_critic_2)

        return self.actor, self.critic_1


    def update_target(self, current, target):
        current_weights = current.weights
        target_weights = target.weights
        for (a, b) in zip(target_weights, current_weights):
            a.assign(a * self.polyak + b * (1 - self.polyak))
       # updated_weights = self.polyak * target_weights + (1-self.polyak) * current_weights
       # target.set_weights(updated_weights)

    def policy(self, state): # might be a bit geared to current GrowthModel()
        action = self.actor(state)
        action += np.random.normal(0, 2, size= action.shape)
        final_act = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return final_act

class DDPG:
    pass

# Generally, should create problem specific actors

def create_actor(state_size, action_size, number_units = 32, activation='relu'):
    actor = tf.keras.models.Sequential([
        layers.Dense(number_units, input_shape=(state_size,), activation=activation),
        layers.Dense(number_units, activation=activation),
        layers.Dense(number_units, activation=activation),
        layers.Dense(action_size)
    ])
    return actor

def create_critic(state_size, action_size, number_units=32, activation='relu'):
    state_input = layers.Input(shape=(state_size,))
    out1 = layers.Dense(number_units, activation=activation)(state_input)
    out1 = layers.Dense(number_units, activation=activation)(out1)


    action_input = layers.Input(shape=(action_size,))
    out2 = layers.Dense(number_units, activation=activation)(action_input)
    out2 = layers.Dense(number_units, activation=activation)(out2)

    concat = layers.Concatenate()([out1, out2])
    out3 = layers.Dense(number_units, activation=activation)(concat)
    out3 = layers.Dense(number_units, activation=activation)(out3)
    final_out = layers.Dense(1)(out3)

    return tf.keras.models.Model([state_input, action_input], final_out)



