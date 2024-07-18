import numpy as np
import os
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
# from torchviz import make_dot
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import trange, tqdm
from scipy.optimize import minimize
from matplotlib import rcParams

import csv
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_tensor_type('torch.DoubleTensor')
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from tqdm import trange, tqdm
from matplotlib import rcParams
import sys
if sys.platform == 'win32':
    NUM_WORKERS = 0 # Windows does not support multiprocessing
else:
    NUM_WORKERS = 2
from tqdm import trange
import random

config = {
        "font.family":'serif',
        "font.size": 10,
        "mathtext.fontset":'stix',
        "font.serif": ['SimSun'],
}
rcParams.update(config)
np.random.seed(42)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F

class ufunc(nn.Module):
    def __init__(self):
        super(ufunc, self).__init__()
        self.fc1 = nn.Linear(11, 100)
        self.fc2 = nn.Linear(11, 100)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 4)
    def forward(self, x):
        #x=x.to(torch.float32)
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x))
        x3 = torch.cat((x1, x2), dim=1)
        x = self.fc3(x3)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def sat(a, maxv):
    n = np.linalg.norm(a)
    if n > maxv:
        return a / n * maxv
    else:
        return a

def sat(a, maxv):
    n = np.linalg.norm(a)
    if n > maxv:
        return a / n * maxv
    else:
        return a

def get_delta_v(x):
    # 从输入状态量提取无人机加速度，速度，角度，角速度和图像跟踪误差量
#     output_acc_NED = x[0:3] # 加速度
#     uavAngRate = x[3] # 角速度
#     uavVelNED = x[4:7] # 速度
#     uavAngEular = x[7:10] # 旋转角度
#     delta_uv = x[10:12] # 图像跟踪误差
    uavVelNED = x[0:3] # 速度
    uavAngEular = x[3:6] # 旋转角度
    delta_uv = x[6:8] # 图像跟踪误差

    # 机体相机坐标系到机身坐标系的旋转矩阵
    R_cb = np.array([[1,0,0],\
                         [0,0,1],\
                         [0,-1,0]])
    
    # 相机系下的目标方向矢量，即一直冲向中心的前方
    n_cc = np.array([0,0,1])

    # 求解方向的变换为机身坐标系下的目标方向矢量
    n_bc = R_cb.dot(n_cc)

    # 基于欧拉角求解四元数
    cy = np.cos(uavAngEular[0] * 0.5)
    sy = np.sin(uavAngEular[0] * 0.5)
    cp = np.cos(uavAngEular[1] * 0.5)
    sp = np.sin(uavAngEular[1] * 0.5)
    cr = np.cos(uavAngEular[2] * 0.5)
    sr = np.sin(uavAngEular[2] * 0.5)

    q0 = cy * cp * cr + sy * sp * sr
    q1 = cy * cp * sr - sy * sp * cr
    q2 = sy * cp * sr + cy * sp * cr
    q3 = sy * cp * cr - cy * sp * sr

    # 基于四元数求解旋转矩阵
    R_ae = np.array([[q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                  [2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],
                  [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2-q1**2-q2**2+q3**2]])
    # R_ba = np.array([[0,1,0], [-1,0,0], [0,0,1]]) #mavros_coordinate to body_coordinate
    # 将ROS坐标系转换为惯性系
    R_ba = np.array([[0,1,0], [-1,0,0], [0,0,1]]) #mavros_coordinate to baselink_coordinate  // body to enu  # body: right-front-up3rpos_est_body)

    # 机体相机坐标系到机身坐标系的旋转矩阵
    R_cb = np.array([[1,0,0],\
                         [0,0,1],\
                         [0,-1,0]])

    # 机体旋转矩阵转换为baselink坐标系下的
    mav_R = R_ae.dot(R_ba)

    n_ec = mav_R.dot(n_bc)
    
    #calacute the no
    # 求解图像误差向量，三个维度分别为
    n_co = np.array([delta_uv[0], delta_uv[1], 370.0])
    # 归一化处理
    n_co = n_co / np.linalg.norm(n_co)
    # 转换为机体坐标系下的误差矢量
    n_bo = R_cb.dot(n_co)
    n_eo = mav_R.dot(n_bo)

    # 两种用法：1）给定世界系下固定的n_td，限定打击方向；2）相对光轴一向量，随相机运动
    n_td = np.array([0, 1, 0], dtype=np.float64)
    n_td = np.array([np.cos(uavAngEular[0]), np.sin(uavAngEular[0]), 0.], dtype=np.float64)
    # n_td = n_ec
    # n_td /= np.linalg.norm(n_td)
    # print("delta_image:",n_eo - n_td)
    v_1 = 2.0 * (n_eo - n_td)   # n_t -> n_td
    v_2 = 1.0 * n_td            # v   -> n_td

    v_d = v_1 + v_2
    v_d = v_d / np.linalg.norm(v_d)
    V = np.linalg.norm(uavVelNED)
    # v_d *= min(V + 2.5, 12)
    v_d = v_d*(V + 2)
    a_d = sat(1.0 * (v_d - uavVelNED), 6.)
    delta_v = v_d - uavVelNED #+ np.array([0., 0., -0.5])
    # a_d = 1.5 * (v_d - pos_info["mav_vel"]) #+ np.array([0., 0., -0.5])
    yaw_rate = -0.002*delta_uv[0]
    return np.array([a_d[0], a_d[1], a_d[2], yaw_rate])
#     return delta_v
#     return delta_v

def get_delta_neo_ntd(x):
    # 从输入状态量提取无人机加速度，速度，角度，角速度和图像跟踪误差量
#     output_acc_NED = x[0:3] # 加速度
#     uavAngRate = x[3] # 角速度
#     uavVelNED = x[4:7] # 速度
#     uavAngEular = x[7:10] # 旋转角度
#     delta_uv = x[10:12] # 图像跟踪误差
    uavVelNED = x[0:3] # 速度
    uavAngEular = x[3:6] # 旋转角度
    delta_uv = x[6:8] # 图像跟踪误差

    # 机体相机坐标系到机身坐标系的旋转矩阵
    R_cb = np.array([[1,0,0],\
                         [0,0,1],\
                         [0,-1,0]])
    
    # 相机系下的目标方向矢量，即一直冲向中心的前方
    n_cc = np.array([0,0,1])

    # 求解方向的变换为机身坐标系下的目标方向矢量
    n_bc = R_cb.dot(n_cc)

    # 基于欧拉角求解四元数
    cy = np.cos(uavAngEular[0] * 0.5)
    sy = np.sin(uavAngEular[0] * 0.5)
    cp = np.cos(uavAngEular[1] * 0.5)
    sp = np.sin(uavAngEular[1] * 0.5)
    cr = np.cos(uavAngEular[2] * 0.5)
    sr = np.sin(uavAngEular[2] * 0.5)

    q0 = cy * cp * cr + sy * sp * sr
    q1 = cy * cp * sr - sy * sp * cr
    q2 = sy * cp * sr + cy * sp * cr
    q3 = sy * cp * cr - cy * sp * sr

    # 基于四元数求解旋转矩阵
    R_ae = np.array([[q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                  [2*(q1*q2+q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3-q0*q1)],
                  [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0**2-q1**2-q2**2+q3**2]])
    # R_ba = np.array([[0,1,0], [-1,0,0], [0,0,1]]) #mavros_coordinate to body_coordinate
    # 将ROS坐标系转换为惯性系
    R_ba = np.array([[0,1,0], [-1,0,0], [0,0,1]]) #mavros_coordinate to baselink_coordinate  // body to enu  # body: right-front-up3rpos_est_body)

    # 机体相机坐标系到机身坐标系的旋转矩阵
    R_cb = np.array([[1,0,0],\
                         [0,0,1],\
                         [0,-1,0]])

    # 机体旋转矩阵转换为baselink坐标系下的
    mav_R = R_ae.dot(R_ba)

    n_ec = mav_R.dot(n_bc)
    
    #calacute the no
    # 求解图像误差向量，三个维度分别为
    n_co = np.array([delta_uv[0], delta_uv[1], 370.0])
    # 归一化处理
    n_co = n_co / np.linalg.norm(n_co)
    # 转换为机体坐标系下的误差矢量
    n_bo = R_cb.dot(n_co)
    n_eo = mav_R.dot(n_bo)

    # 两种用法：1）给定世界系下固定的n_td，限定打击方向；2）相对光轴一向量，随相机运动
    # n_td = np.array([0, 1, 0], dtype=np.float64)
    n_td = np.array([np.cos(uavAngEular[0]), np.sin(uavAngEular[0]), 0.], dtype=np.float64)
    return n_eo,n_td

if __name__ == "__main__":

    random.seed(42)
    random.seed(42)
    get_data_list = []
    output_acc_list = []
    n_eo_list = []
    n_td_list = []

    vx_min = -2.0
    vx_max = 2.0
    vy_min = -15.0
    vy_max = 0.0
    vz_min = -2.0
    vz_max = 2.0
    yaw_min = -3.14
    yaw_max = 3.14
    pitch_min = -0.1
    pitch_max = 0.2
    roll_min = -0.25
    roll_max = 0.25
    du_min = -640.0
    du_max = 640.0
    dv_min = -360.0
    dv_max = 360.0
    min_list = [vx_min,vy_min,vz_min,yaw_min,pitch_min,roll_min,du_min,dv_min]
    max_list = [vx_max,vy_max,vz_max,yaw_max,pitch_max,roll_max,du_max,dv_max]

    for i in trange(20000):
        data_list_one = []
        for i in range(6):
            data_list_one.append(random.uniform(min_list[i],max_list[i]))
        for i in range(6,8):
            data_list_one.append(random.uniform(min_list[i],max_list[i]))
        data_list_one = np.array(data_list_one)
        get_data_list.append(data_list_one)
        data_list_two = data_list_one.copy()
    #     data_list_two[1] = -1.0*data_list_two[1]
        output_acc = get_delta_v(data_list_two)
        n_eo,n_td = get_delta_neo_ntd(data_list_two)
        n_eo_list.append(n_eo)
        n_td_list.append(n_td)
        output_acc_list.append(output_acc)

    # plt.scatter(get_data_list[:,0],get_data_list[:,1])
    vx_min = -2.0
    vx_max = 2.0
    vy_min = -2.0
    vy_max = 2.0
    vz_min = -2.0
    vz_max = 2.0
    yaw_min = -2.5
    yaw_max = -1.0
    pitch_min = -0.1
    pitch_max = 0.2
    roll_min = -0.25
    roll_max = 0.25
    du_min = -640.0
    du_max = 640.0
    dv_min = -360.0
    dv_max = 360.0
    min_list = [vx_min,vy_min,vz_min,yaw_min,pitch_min,roll_min,du_min,dv_min]
    max_list = [vx_max,vy_max,vz_max,yaw_max,pitch_max,roll_max,du_max,dv_max]
    for i in trange(10000):
        data_list_one = []
        for i in range(6):
            data_list_one.append(random.uniform(min_list[i],max_list[i]))
        for i in range(6,8):
            data_list_one.append(random.uniform(min_list[i],max_list[i]))
        data_list_one = np.array(data_list_one)
        get_data_list.append(data_list_one)
        data_list_two = data_list_one.copy()
    #     data_list_two[1] = -1.0*data_list_two[1]
        output_acc = get_delta_v(data_list_two)
        n_eo,n_td = get_delta_neo_ntd(data_list_two)
        n_eo_list.append(n_eo)
        n_td_list.append(n_td)
        output_acc_list.append(output_acc)
    get_data_list = np.vstack(get_data_list)
    output_acc_list = np.vstack(output_acc_list)
    n_eo_list = np.vstack(n_eo_list)
    n_td_list = np.vstack(n_td_list)
    print(get_data_list.shape)
    print(output_acc_list.shape)
    print(n_eo_list.shape)
    print(n_td_list.shape)
    input_list = np.hstack((n_eo_list,n_td_list,get_data_list[:,0:3],get_data_list[:,6:8]))
    torch.manual_seed(42)
    # 导入之前的D函数神经网络，并固定权重

    # 设置控制器神经网络的结构
    k_net= ufunc()
    # k_net= torch.load('k_net_cpu_still_NEO_NTD_3.pt')
    # 损失函数和优化器设置
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(k_net.parameters(),lr =1e-3,betas = (0.9,0.999))
    # 设置策略更新损失函数的求解目标
    vae_eval = -0
    u_train_loss = np.array([])
    a_list = []
    # 训练次数不宜过多，否则容易出现过拟合的问题
    for epoch in trange(100000):
        u_label = torch.from_numpy(output_acc_list)
        get_data_list_2 = input_list.copy()
        get_data_list_2[:,9:11] = get_data_list_2[:,9:11] / 370.0
        x_feature = torch.from_numpy(get_data_list_2)
        # 控制器神经网络输出结果
        control_output = k_net(x_feature)
        # 更新约束损失函数
        loss2 = criterion(control_output.to(torch.float32),u_label.to(torch.float32))

        # print('loss1',loss1.detach().numpy())
        # print('loss2',loss2.detach().numpy())

        loss = 10000*loss2
        # 存储训练过程的损失函数变化
        u_train_loss=np.append(u_train_loss,loss.detach().numpy())
        # 更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
    torch.save(k_net,'./k_net_cpu_still_NEO_NTD_5.pt')