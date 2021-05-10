import numpy as np
import os


def main(beta, adv, agent):
    data_path = os.path.join('./model/simple_tag_adv_qmix_agent_maddpg_'
                             + str(beta) + '_social_' + ('adv' if adv else '') + (
                                 'agent' if agent else ''))

    adv_data = np.load(os.path.join(data_path, 'adversary/qmix/data/adversary0_returns.pkl.npy'))
    agent_data = np.load(os.path.join(data_path, 'agent/maddpg/data/agent3_returns.pkl.npy'))

    print(beta, 'adv' if adv else '', 'agent' if agent else '')
    print('adv:', np.max(adv_data))
    print('agent:', np.min(agent_data))


if __name__ == '__main__':
    # main(-5.0, True, False)
    # main(-1.581, True, False)
    # main(-0.5, True, False)
    # main(-0.158, True, False)
    # main(-0.05, True, False)
    # main(0.05, True, False)
    # main(0.5, True, False)

    main(-5.0, False, True)
    main(-0.5, False, True)
    main(-0.05, False, True)
    main(0.05, False, True)
    main(0.5, False, True)
