import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LSTMDataset
from model import LSTMModel

if __name__ == "__main__":
    # 数据准备
    seq_len = 2
    data_file = './data.txt'
    dataset = LSTMDataset(data_file, seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # 模型参数设置
    input_size = 2
    hidden_size = 64
    num_layers = 1
    output_size = 1
    batch_size = 2

    # 模型初始化
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, batch_size)

    # 损失函数和优化器设置
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    truth = []
    predict = []
    # 模型训练
    num_epochs = 1
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.unsqueeze(-1)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            truth += labels.tolist()
            predict += outputs.tolist()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, len(dataloader),
                                                                         loss.item()))

    import matplotlib.pyplot as plt

    # 绘制真实值散点图
    plt.scatter(range(len(truth)), truth, c='blue', label='Truth')

    # 绘制预测值散点图
    plt.scatter(range(len(predict)), predict, c='red', label='Predict')

    # 添加标题和坐标轴标签
    plt.title("Truth-Predict Scatter Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")

    # 添加图例
    plt.legend()

    # 显示图像
    plt.show()

    truth = []
    predict = []
    model.eval()
    data_file = './test.txt'
    dataset = LSTMDataset(data_file, seq_len)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.unsqueeze(-1)
            outputs = model(inputs)
            predict += outputs.tolist()
            truth += labels.tolist()
    # 绘制真实值散点图
    plt.scatter(range(len(truth)), truth, c='blue', label='Truth')

    # 绘制预测值散点图
    plt.scatter(range(len(predict)), predict, c='red', label='Predict')

    # 添加标题和坐标轴标签
    plt.title("Truth-Predict Scatter Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")
    # 添加图例
    plt.legend()

    # 显示图像
    plt.show()
