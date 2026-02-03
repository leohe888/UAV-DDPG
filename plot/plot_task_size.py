import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 路径配置 =====
ddpg_dir = r'..\data\ddpg\TaskSize'
dsact_dir = r'..\data\dsact\TaskSize'

user_nums = [60, 80, 100]

def smooth(y, window=10):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')

for u in user_nums:
    plt.figure(figsize=(9, 5))

    # ========= DDPG =========
    ddpg_path = os.path.join(ddpg_dir, f'{u}.pkl')
    with open(ddpg_path, 'rb') as f:
        ddpg_rewards = pickle.load(f)

    ddpg_rewards_s = smooth(ddpg_rewards, window=10)
    ddpg_x = range(1, len(ddpg_rewards_s) + 1)

    plt.plot(ddpg_x, ddpg_rewards_s, label='DDPG', linestyle='-')

    # ========= DSACT =========
    dsact_path = os.path.join(dsact_dir, f'{u}.csv')
    df = pd.read_csv(dsact_path)

    dsact_rewards = df['Value'].values
    dsact_steps = df['Step'].values

    dsact_rewards_s = smooth(dsact_rewards, window=10)
    dsact_x = dsact_steps[:len(dsact_rewards_s)]

    plt.plot(dsact_x, dsact_rewards_s, label='DSACT', linestyle='--')

    # ========= 图像设置 =========
    plt.xlabel('Episode (DDPG) / Step (DSACT)')
    plt.ylabel('Reward')
    plt.title(f'Reward Comparison (TaskSize = {u}Mbits)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()