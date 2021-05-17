# Toward Action-value Function Estimation with Social Influence and Communication in Multi-Agent Reinforcement Learning
This is the project repository for CMU Introduction to Deep Learning Spring 2021. 

Project Abstract: In this project, we focus on multi-agent reinforcement learning (MARL) with continuous action space. Commonly known as the credit assignment problem, one of the fundamental objectives in MARL is to evaluate individual agent's contribution to the joint reward. To address this problem, we present a multi-agent actor-critic algorithm that estimates the joint-action value function as a non-linear combination of per-agent action-value, without imposing additional constraints. Additionally, we apply social influence, as a form of intrinsic motivation and implicit communication, as well as explicit communication, to exploit information sharing in an competitive setting. We empirically evaluate our proposals on a partially observable multi-agent game with continuous action space and demonstrate that our proposals can outperform the state-of-the-art benchmark algorithm.


# Requirements:
1. PyTorch
2. OpenAI Gym: https://github.com/openai/gym
3. OpenAI Multi-Particle Environment: https://github.com/openai/multiagent-particle-envs

# Sources:
1. Baseline (MADDPG) implementation reference : https://github.com/shariqiqbal2810/maddpg-pytorch
2. MADDPG paper: https://arxiv.org/abs/1706.02275
3. QMIX paper: https://arxiv.org/abs/1803.11485

# Run:

`python main.py`.

# Branches:
- The `main` branch is composed by Tony Huang (runqih@andrew.cmu.edu). It contains both the MADDPG baseline and QMIX, and is used as the cornerstone in other branches.
- The `zongyuez` branch is composed by Zongyue Zhao (zongyuez@andrew.cmu.edu). It applies social influence as a form of intrinsic motivation.
- The `jiayu` branch is composed by Jiayu Chang (jiayuc@andrew.cmu.edu). It applies MATD3.
- The `cnariset` branch is composed by Chaitanya Prasad Narisetty (cnariset@andrew.cmu.edu). It applies communcation network.
