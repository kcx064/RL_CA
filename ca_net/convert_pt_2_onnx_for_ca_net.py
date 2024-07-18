import onnx
import torch
import torch.nn

if __name__ ==('__main__'):
    ca_net = torch.load('./ca.pt').to('cpu')
    ca_net.eval()
    print(ca_net)

    batch_size = 1  # 或者任何你想要的批量大小
    dummy_input = torch.randn(batch_size, 4)  # 创建一个形状为 (batch_size, 4) 的随机张量
    # 使用torch.onnx.export导出模型
    torch.onnx.export(ca_net,                # 模型或模型的模块
        dummy_input,                        # 模型输入（或模型输入的元组）
        "ca_model.onnx",                  	# 输出的ONNX文件名
        export_params=True,                 # 存储训练好的参数权重在内
        opset_version=11,                   # 选择适合你的PyTorch版本的操作集版本
        do_constant_folding=True,           # 是否执行常量折叠优化
        input_names = ['input_vector'],     # 输入名称
        output_names = ['output_vector'])   # 输出名称