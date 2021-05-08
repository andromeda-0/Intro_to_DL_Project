import numpy as np


def main():
    data = np.load(
            'C:/Users/Zongyue/OneDrive/Documents/CMU/21Spring/11785/Project/Intro_to_DL_project/model/simple_tag_adv_qmix_agent_maddpg_-0.5_social_adv/agent/maddpg/data/agent3_returns.pkl.npy')

    print(np.min(data))


if __name__ == '__main__':
    main()
