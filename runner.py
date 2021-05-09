import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.replay_buffer import ReplayBuffer
from algorithm.policy import Policy
from constants import device
from tqdm import trange


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.n_agents = len(self.env.agents)

        self.noise = self.args.noise_rate
        self.min_noise = self.args.min_noise_rate
        self.anneal_noise = (self.noise - self.min_noise) / self.args.anneal_episodes
        self.noise_adv = self.args.noise_rate_adv
        self.min_noise_adv = self.args.min_noise_rate_adv
        self.anneal_noise_adv = (self.noise_adv - self.min_noise_adv) / self.args.anneal_episodes

        self.epsilon = self.args.epsilon
        self.min_epsilon = self.args.min_epsilon
        self.anneal_epsilon = (self.epsilon - self.min_epsilon) / self.args.anneal_episodes
        self.epsilon_adv = self.args.epsilon_adv
        self.min_epsilon_adv = self.args.min_epsilon_adv
        self.anneal_epsilon_adv = (self.epsilon_adv - self.min_epsilon_adv) / self.args.anneal_episodes

        self.buffer = ReplayBuffer(self.args, self.env)
        self.policy = Policy.init_from_env(self.args, self.env)
        self.agents = self.policy.agents
        self.mixers = self.policy.mixers
        self.n_teams = self.policy.n_teams
        self.n_agents = self.policy.n_agents
        self.teams = self.policy.teams
        self.save_path = os.path.join(self.args.model_dir, self.args.scenario_name)

    def run(self):
        returns = [[] for _ in range(self.n_agents)]
        norm_scores = [[] for _ in range(self.n_agents)]

        for episode in trange(self.args.n_episodes):
            s = self.env.reset()
            for time_step in range(self.args.episode_length):
                r = []
                with torch.no_grad():
                    torch_obs = [torch.tensor(s[i], dtype=torch.float32, device=device).view(1, -1) for i in
                                 range(self.n_agents)]
                    u = self.policy.step(torch_obs, epsilon=[self.epsilon_adv] * 3 + [self.epsilon],
                                         noise_rate=[self.noise_adv] * 3 + [self.noise])
                # actions = [action.numpy().flatten() for action in u]
                u = actions = torch.cat(u, dim=0).cpu().numpy()
                s_next, rewards, done, _ = self.env.step(actions)
                for i in range(self.n_agents):
                    r.append([rewards[i]])
                self.buffer.store_transition(s, s_next, u, r)
                s = s_next
                if self.buffer.current_size >= self.args.batch_size:
                    update_policy = (time_step % self.args.policy_update_freq) == 0

                    for i, a in enumerate(self.agents):
                        if self.policy.agent_algo[i] == 'maddpg':
                            batch = self.buffer.sample(self.args.batch_size)
                            self.policy.maddpg_update(batch, i)
                        elif self.policy.agent_algo[i] == 'matd3':
                            batch = self.buffer.sample(self.args.batch_size)
                            self.policy.matd3_update(batch, i, update_policy)

                    for i, m in enumerate(self.mixers):
                        if self.policy.team_algo[i] == 'qmix':
                            batch = self.buffer.sample(self.args.batch_size)
                            self.policy.qmix_update(batch, i)
                        elif self.policy.team_algo[i] == 'qmix_td3':
                            batch = self.buffer.sample(self.args.batch_size)
                            self.policy.double_qmix_update(batch, i, update_policy)

                    self.policy.soft_update_non_td3_target_networks()
                    if update_policy:
                        self.policy.soft_update_td3_target_networks()
            self.noise = max(self.min_noise, self.noise - self.anneal_noise)
            self.epsilon = max(self.min_epsilon, self.epsilon - self.anneal_epsilon)

            self.noise_adv = max(self.min_noise_adv, self.noise_adv - self.anneal_noise_adv)
            self.epsilon_adv = max(self.min_epsilon_adv, self.epsilon_adv - self.anneal_epsilon_adv)

            if episode > 0 and episode % self.args.evaluate_rate == 0:
                print('Training Episode = %s' % episode)
                ave_return, norm_return = self.evaluate(render=False)

                for i, a in enumerate(self.agents):
                    agent_num = a.type + str(i)
                    returns[i].append(ave_return[i])
                    data_path = os.path.join(self.save_path, '%s' % a.type,
                                             self.policy.agent_algo[i], 'data')
                    plot_path = os.path.join(self.save_path, '%s' % a.type,
                                             self.policy.agent_algo[i], 'plots')
                    if not os.path.exists(data_path):
                        os.makedirs(data_path)
                    if not os.path.exists(plot_path):
                        os.makedirs(plot_path)
                    np.save(data_path + '/%s_returns.pkl' % agent_num, returns[i])
                    np.save(data_path + '/%s_norm_score.pkl' % agent_num, norm_scores[i])
                    plt.figure()
                    plt.plot(range(len(returns[i])), returns[i])
                    plt.xlabel('episode * ' + str(self.args.evaluate_rate))
                    plt.ylabel('average_returns')
                    plt.title('%s training (%s)' % (agent_num, self.args.scenario_name))
                    plt.savefig(plot_path + '/%s_returns.png' % agent_num)

        print('Complete Training')
        self.policy.save_model()
        ave_return, norm_return = self.evaluate(render=False)

    @torch.no_grad()
    def evaluate(self, render=False):
        returns = [[] for _ in range(self.n_agents)]
        norm_return = [[] for _ in range(self.n_agents)]

        for episode in trange(self.args.evaluate_episodes):
            s = self.env.reset()
            r_e = [0 for _ in range(self.n_agents)]
            for time_step in range(self.args.episode_length):
                if render:
                    self.env.render()
                torch_obs = [torch.tensor(s[i], dtype=torch.float32, device=device).view(1, -1) for i in
                             range(self.n_agents)]
                u = self.policy.step(torch_obs, epsilon=0,
                                     noise_rate=[.05] * 3 + [0])  # only predator use noise during eval
                actions = torch.cat(u, dim=0).cpu().numpy()
                # actions = [action.cpu().numpy().flatten() for action in u]
                s_next, r, done, info = self.env.step(actions)
                s = s_next
                for i in range(self.n_agents):
                    r_e[i] += r[i]
            for i in range(self.n_agents):
                returns[i].append(r_e[i])

        # for i in range(self.args.evaluate_episodes):
        #     for j in range(self.n_agents):
        #         norm_return[j].append(
        #             (returns[j][i] - min(returns[j])) / (max(returns[j]) - min(returns[j])))

        for j in range(self.n_agents):
            min_j = min(returns[j])
            max_j = max(returns[j])
            for i in range(self.args.evaluate_episodes):
                norm_return[j].append((returns[j][i] - min_j) / (max_j - min_j))

        ave_return = [np.mean(i) for i in returns]
        norm_score = [np.mean(i) for i in norm_return]
        for i, a in enumerate(self.agents):
            print('{} {}: max = {:.3f}, min = {:.3f}, mean = {:.3f}, mean norm = {:.3f}'.
                  format(a.type, i, max(returns[i]), min(returns[i]), ave_return[i], norm_score[i]))

        return ave_return, norm_score
