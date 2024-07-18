import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

class ufunc3(nn.Module):
    def __init__(self):
        super(ufunc3, self).__init__()
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
    
if __name__ ==('__main__'):
    k_net = torch.load('./k_net_cpu_still_NEO_NTD_5.pt').to('cpu')
    k_net.eval()
    print(k_net)

    batch_size = 1  # 或者任何你想要的批量大小
    dummy_input = torch.randn(batch_size, 11, dtype=torch.float64)  # 创建一个形状为 (batch_size, 4) 的随机张量
    # 使用torch.onnx.export导出模型
    torch.onnx.export(k_net,                # 模型或模型的模块
        dummy_input,                        # 模型输入（或模型输入的元组）
        "k_net.onnx",                  	# 输出的ONNX文件名
        export_params=True,                 # 存储训练好的参数权重在内
        opset_version=11,                   # 选择适合你的PyTorch版本的操作集版本
        do_constant_folding=True,           # 是否执行常量折叠优化
        input_names = ['input_vector'],     # 输入名称
        output_names = ['output_vector'])   # 输出名称