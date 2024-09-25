import random

class fzenv:
    def __init__(self):
        self.pitch_t = 20*random.uniform(-1, 1)
        self.yaw_t = 20*random.uniform(-1, 1)
        self.pitch_frame = self.pitch_t
        self.yaw_frame = self.yaw_t
        self.done=False

        self.reward = 0
        self.step_counter = 0

    def reset(self):
        self.pitch_t = 20*random.uniform(-1, 1)
        self.yaw_t = 20*random.uniform(-1, 1)
        self.pitch_frame = self.pitch_t
        self.yaw_frame = self.yaw_t
        self.done=False
        
        self.reward = 0
        self.step_counter = 0
        return self.pitch_frame, self.yaw_frame

    def step(self, action):
        # 
        pitch = action[0][0]
        yaw = action[0][1]
        # store old values
        self.pitch_frame_old = self.pitch_frame
        self.yaw_frame_old = self.yaw_frame
        # update values
        self.pitch_frame = self.pitch_frame - pitch
        self.yaw_frame = self.yaw_frame - yaw
        
        next_state = [self.pitch_frame, self.yaw_frame]

        self.step_counter +=1
        # 计算奖励方式1  初步训练可将0.1改为0，避免稀疏奖励，初次训练可以增加噪声方差至1，刺激学习，稳定后可降至0.01
        if (abs(self.pitch_frame) - abs(self.pitch_frame_old) < 0.1) or (abs(self.yaw_frame) - abs(self.yaw_frame_old) < 0.1):
            self.reward = 1
            if (abs(self.pitch_frame) - abs(self.pitch_frame_old) < 0.1) and (abs(self.yaw_frame) - abs(self.yaw_frame_old) < 0.1):
                self.reward = self.reward + 2
            # if abs(self.pitch_frame)<1 and abs(self.yaw_frame)<1:
            #     self.reward += 5
        else:
            # self.reward = self.reward - 1 # 首先设置0.1，然后训练一定次数后惩罚增加至1
            self.reward = -10

        # 计算奖励方式2
        # reward_f = - (abs(self.pitch_frame) - abs(self.pitch_frame_old)) - (abs(self.yaw_frame) - abs(self.yaw_frame_old))
        # self.reward = int(reward_f)

        # # 计算奖励3
        # if abs(self.pitch_frame) - abs(self.pitch_frame_old) < -0.1:
        #     self.reward = self.reward + 1
        # else:
        #     self.reward = self.reward - 5
        # if abs(self.yaw_frame) - abs(self.yaw_frame_old) < -0.1:
        #     self.reward = self.reward + 1
        # else:
        #     self.reward = self.reward - 5

        # 每多一步，奖励减1
        # self.reward = self.reward - 1

        if abs(self.pitch_frame)<0.5 and abs(self.yaw_frame)<0.5:
            self.done=True
            # print("Reached goal")
            self.done_result = 3
        else: 
            if self.step_counter>500:
                self.done=True
                # print("Max steps reached")
                self.done_result = 2
            else: 
                self.done=False
                self.done_result = 0
                # if abs(self.pitch_frame)>30 or abs(self.yaw_frame)>30:
                #     self.done=True
                #     # print("Out of bounds")
                #     self.done_result = 1
                # else:
                #     self.done=False
                #     self.done_result = 0


        return next_state, self.reward, self.done, self.step_counter, self.done_result