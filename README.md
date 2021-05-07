# CMU Introduction to Deep Learning Spring 2020 Team Project - Social Influence

This branch is contains all work related to social influence in this project.

## Requirements:

- `Python == 3.9`
- `PyTorch == 1.8.1+cu111`
- `OpenAI Gym == 0.10.5`: https://github.com/openai/gym `
- `OpenAI Multi-Particle Environment`: https://github.com/openai/multiagent-particle-envs

Install the packages specified above, then replace the `environment.py` in the installed package with `environment/environment.py` in this branch. The only modification in this script is in the `__init__()` function, for the compatibility with a continous action space.

## Sources:

1. Baseline implementation (MADDPG): https://github.com/shariqiqbal2810/maddpg-pytorch
2. MADDPG paper: https://arxiv.org/abs/1706.02275
3. QMIX paper: https://arxiv.org/abs/1803.11485
4. Social influence paper: http://proceedings.mlr.press/v97/jaques19a/jaques19a.pdf

## Run:

Please check `misc.py` for all supported arguments. To reproduce the best result we reported on social influence, please run `python main.py --social_adv`.
