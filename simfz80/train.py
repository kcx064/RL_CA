import numpy as np
import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
# import gym
from myenv import fzenv
from parsers import args
from DDPGmodel import ReplayBuffer, DDPG
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -------------------------------------- #
# 创建图片保存文件夹
# -------------------------------------- #
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 构造文件夹名称，这里我们简单地将时间作为文件夹名
folder_name = current_time

# 构造完整的文件夹路径（这里假设我们想要在当前工作目录下创建这个文件夹）
folder_path = os.path.join(os.getcwd(), folder_name)

# 创建文件夹
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# -------------------------------------- #
# 环境加载
# -------------------------------------- #
 
env_name = "MountainCarContinuous-v0"  # 连续型动作
# env = gym.make(env_name, render_mode="human")
env = fzenv()
n_states = 2
n_actions = 2
action_bound = 1.0  # 动作的最大值 1.0

 
# -------------------------------------- #
# 模型构建
# -------------------------------------- #
 
# 经验回放池实例化
replay_buffer = ReplayBuffer(capacity=args.buffer_size)
# 模型实例化
agent = DDPG(n_states = n_states,  # 状态数
             n_hiddens = args.n_hiddens,  # 隐含层数
             n_actions = n_actions,  # 动作数
             action_bound = action_bound,  # 动作最大值
             sigma = args.sigma,  # 高斯噪声
             actor_lr = args.actor_lr,  # 策略网络学习率
             critic_lr = args.critic_lr,  # 价值网络学习率
             tau = args.tau,  # 软更新系数
             gamma = args.gamma,  # 折扣因子
             device = device,
             filename = args.filename # 模型保存名字前缀
            )
 
# -------------------------------------- #
# 模型训练
# -------------------------------------- #
 
return_list = []  # 记录每个回合的return
mean_return_list = []  # 记录每个回合的return均值
step_count_list = []  # 记录每个回合的step
done_result_list = []  # 记录每个回合的done原由
action_pitch_list = []  # 记录每个回合的action
action_yaw_list = [] 
 
for i in range(1000):  # 迭代10回合
    episode_return = 0  # 累计每条链上的reward
    state = env.reset()  # 初始时的状态
    # state[1] = env.reset()[1]  # 初始时的状态
    done = False  # 回合结束标记
    trained = False
 
    while not done:
        # 获取当前状态对应的动作
        action = agent.take_action(state)
        action_pitch_list.append(action[0][0])
        action_yaw_list.append(action[0][1])
        # 环境更新
        next_state, reward, done, step_count, done_result = env.step(action)
        # 更新经验回放池
        replay_buffer.add(state, action, reward, next_state, done)
        # 状态更新
        state = next_state
        # 累计每一步的reward
        episode_return += reward
        # print(f'reward:{done},state:{state},action:{action},reward:{reward}')
        # print(f'action:{action},reward:{reward}')
 
        # 如果经验池超过容量，开始训练
        if (replay_buffer.size() > args.min_size) and args.train_mode:
            trained = True
            # 经验池随机采样batch_size组
            s, a, r, ns, d = replay_buffer.sample(args.batch_size)
            # 构造数据集
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
                'dones': d,
            }
            # 模型训练
            agent.update(transition_dict)
    
    # 保存每一个回合的回报
    return_list.append(episode_return)
    step_count_list.append(step_count)
    mean_return_list.append(np.mean(return_list[-10:]))  # 平滑
    done_result_list.append(done_result)
    # 打印回合信息
    # print(f'iter:{i},done{done_result},episode:{episode_return},reward:{reward},action:{action},state:{state},mean_return:{np.mean(return_list[-10:])}')

    # 显示action曲线
    action_range = list(range(len(action_pitch_list)))
    plt.subplot(211)
    plt.plot(action_range, action_pitch_list)
    plt.subplot(212)
    plt.plot(action_range, action_yaw_list)
    # plt.show()
    image_path = os.path.join(folder_path, f'action{i}.png')
    plt.savefig(image_path)
    plt.clf() # 注释掉这行命令可以获取，所有action的历史曲线
    action_pitch_list.clear()
    action_yaw_list.clear()


agent.save_model(args.filename)
# -------------------------------------- #
# 绘图
# -------------------------------------- #

x_range = list(range(len(return_list)))

plt.subplot(411)
plt.plot(x_range, return_list)  # 每个回合return
plt.xlabel('episode')
plt.ylabel('reward')

plt.subplot(412)
plt.plot(x_range, mean_return_list)  # 每回合return均值
plt.xlabel('episode')
plt.ylabel('mean_reward')

plt.subplot(413)
plt.plot(x_range, step_count_list)  # 每回合return均值
plt.xlabel('episode')
plt.ylabel('step_count')

plt.subplot(414)
plt.plot(x_range, done_result_list)  # 每回合return均值
plt.xlabel('episode')
plt.ylabel('done_result')


# plt.show()
plt.savefig(image_path)