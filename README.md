# Toward Action-value Function Estimation with Social Influence and Communication in Multi-Agent Reinforcement Learning
CMU Introduction to Deep Learning Spring 2021 Team Project

This branch implements MATD3 and combine the ideas in TD3 to QMIX.

## Requirements:

- `Python == 3.9`
- `PyTorch == 1.8.1+cu111`
- `OpenAI Gym == 0.10.5`: https://github.com/openai/gym `
- `OpenAI Multi-Particle Environment`: https://github.com/openai/multiagent-particle-envs

Install the packages specified above, then replace the `environment.py` in the installed package with `environment/environment.py` in branch `zongyuez`. The only modification in the script is in the `__init__()` function, for the compatibility with a continous action space.

## Sources:

1. Baseline implementation (MADDPG): https://github.com/shariqiqbal2810/maddpg-pytorch
2. MADDPG paper: https://arxiv.org/abs/1706.02275
3. QMIX paper: https://arxiv.org/abs/1803.11485
4. TD3 paper https://arxiv.org/abs/1802.09477 

## Run:

Run `python main.py` for training. See `misc.py` for arguments. 
