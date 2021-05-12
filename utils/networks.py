import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64,
                 constrain_out=False, discrete_action=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        if constrain_out and not discrete_action:
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.out_fn(self.fc3(x))
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64,
                 constrain_out=False, discrete_action=True):
        super(LSTM, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        if constrain_out and not discrete_action:
            final_fc = nn.Linear(hidden_dim, output_dim)
            final_fc.weight.data.uniform_(-3e-3, 3e-3)
            self.fc3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     final_fc)
            self.out_fn = torch.tanh
        else:
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.out_fn = lambda x: x

    def forward(self, x, K=2):
        """
        K : number of communication steps
        """
        h = self.fc2(F.relu(self.fc1(x)))
        for k in range(K):
            c = torch.zeros_like(h)
            h = self.gru(c,h)
        out = self.out_fn(self.fc3(h))
        return out
    
class LSTM_COMM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64,
                 constrain_out=False, discrete_action=True):
        super(LSTM_COMM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        if constrain_out and not discrete_action:
            final_fc = nn.Linear(hidden_dim, output_dim)
            final_fc.weight.data.uniform_(-3e-3, 3e-3)
            self.fc3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     final_fc)
            self.out_fn = torch.tanh
        else:
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.out_fn = lambda x: x

    def forward(self, x, agent_ix=0, K=2):
        """
        K : number of communication steps
        """
        if len(x.shape) == 3: n_agents = x.shape[1]
        else: n_agents = 1
        if agent_ix >= n_agents: agent_ix = agent_ix-3 #TODO: adhoc, change to accomodate general case

        h = self.fc2(F.relu(self.fc1(x.view(-1,self.input_dim))))
        c = torch.zeros_like(h)
        h = self.gru(c,h)
        for k in range(1,K):
            h = h.reshape(-1, n_agents, self.hidden_dim)
            c = h.reshape(-1, 1, n_agents*self.hidden_dim)
            c = c.repeat(1, n_agents, 1)
            mask = (1 - torch.eye(n_agents)).to(DEVICE)
            c = c * mask.view(-1, 1).repeat(1, self.hidden_dim).view(n_agents, -1).unsqueeze(0)
            c = c.reshape(-1, n_agents, n_agents, self.hidden_dim).mean(dim=-2)
            h = self.gru(c.reshape(-1, self.hidden_dim),h.reshape(-1, self.hidden_dim))
        h = h.view(-1, n_agents, self.hidden_dim)[:, agent_ix]
        out = self.out_fn(self.fc3(h))
        return out

class QMixer(nn.Module):
    def __init__(self, state_dim, n_agents, output_dim=1,
                 embed_dim=32, hypernet_embed=64):
        super(QMixer, self).__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.n_agents = n_agents

        self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        self.hyper_w2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, self.embed_dim))

        self.hyper_b1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                      nn.ReLU(),
                                      nn.Linear(hypernet_embed, output_dim))

    def forward(self, agent_qs, states):
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)

        hidden_1 = F.relu(torch.bmm(agent_qs, w1) + b1)

        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        y = torch.bmm(hidden_1, w2) + b2
        q_tot = y.view(-1, 1)

        return q_tot
