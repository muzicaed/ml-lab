import numpy as np
import torch as T
from ppo import ActorNet, CriticNet, Memory

POLICY_CLIP = 0.2


class Agent:
    # learning_rate Alpha
    # discount_factor Gamma
    # tradeoff Lamda
    def __init__(self, actions_count, inputs_dim, learning_rate, discount_factor, tradeoff, no_of_epochs=10, batch_size=64):
        self.discount_factor = discount_factor
        self.tradeoff = tradeoff
        self.no_of_epochs = no_of_epochs

        self.actor_net = ActorNet(actions_count, inputs_dim, learning_rate)
        self.critic_net = CriticNet(inputs_dim, learning_rate)
        self.memory = Memory(batch_size)

    def remember(self, state, action, probs, values, reward, done):
        self.memory.store(state, action, probs, values, reward, done)

    def choose_action(self, state):
        state_tensor = T.tensor([state], dtype=T.float).to(self.actor_net.device)
        distribution = self.actor_net(state_tensor)
        value = self.critic_net(state_tensor)
        action = distribution.sample()

        probs = T.squeeze(distribution.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.no_of_epochs):
            states, actions, old_probs, values, rewards, dones, batches = self.memory.create_batches()
            advantage = np.zeros(len(rewards), dtype=np.float32)

            rewards_len = len(rewards) - 1
            for time_step in range(rewards_len):
                discount = 1
                advantage_t = 0
                for k in range(time_step, rewards_len):
                    advantage_t += self.__calc_t_advantage(discount, rewards[k], values[k], values[k+1], dones[k], )
                    discount *= self.discount_factor * self.tradeoff

                advantage[time_step] = advantage_t

            advantage_tensor = T.tensor(advantage).to(self.actor_net.device)
            values_tensor = T.tensor(values).to(self.critic_net.device)

            for batch in batches:
                states = T.tensor(states[batch], dtype=T.float).to(self.actor_net.device)
                old_probs = T.tensor(old_probs[batch]).to(self.actor_net.device)
                actions = T.tensor(actions[batch]).to(self.actor_net.device)

                distribution = self.actor_net(states)
                critic_value = self.critic_net(states)
                critic_value = T.squeeze(critic_value)

                new_probs = distribution.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()
                w_probs = prob_ratio * advantage_tensor[batch]
                w_clipped_probs = T.clamp(prob_ratio, 1 - POLICY_CLIP, 1 + POLICY_CLIP) * advantage_tensor[batch]
                actor_loss = -T.min(w_probs, w_clipped_probs).mean()

                returns = advantage_tensor[batch] + values_tensor[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()
                total_loss = actor_loss + 0.5 * critic_loss

                self.actor_net.optimizer.zero_grad()
                self.critic_net.optimizer.zero_grad()
                total_loss.backward()
                self.actor_net.optimizer.step()
                self.critic_net.optimizer.step()

        self.memory.clear()

    def __calc_t_advantage(self, discount, reward, value, next_value, done):
        return discount * (reward + self.discount_factor * next_value * (1-int(done)) - value)

    def save_models(self):
        self.actor_net.save()
        self.critic_net.save()
        print('Models saved')

    def load_models(self):
        self.actor_net.load()
        self.critic_net.load()
        print('Models loaded')
