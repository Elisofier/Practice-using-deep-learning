import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from tqdm import tqdm

from my_data_set import MyDataset
from my_model import MyModel

data_file = './data.txt'
label_file = './label.txt'
dataset = MyDataset(data_file, label_file, train_frac=0.7, val_frac=0.15, test_frac=0.15, batch_size=32)

train_dataloader = dataset.get_train_dataloader()
val_dataloader = dataset.get_val_dataloader()
test_dataloader = dataset.get_test_dataloader()

train_loss_list = []
val_loss_list = []
test_loss_list = []


# 定义训练函数
def train(the_model, train_data, val_data, num_epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(the_model.parameters(), lr=lr)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        the_model.train()
        train_loss = 0.0
        for X, Y in tqdm(train_data, desc=f'Epoch {epoch + 1} (train)'):
            optimizer.zero_grad()
            pred_y = the_model(X)
            loss = criterion(pred_y, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_data.dataset)
        train_loss_list.append(train_loss)
        the_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in tqdm(val_data, desc=f'Epoch {epoch + 1} (val)'):
                pred_y = the_model(X)
                loss = criterion(pred_y, Y)
                val_loss += loss.item()
            val_loss /= len(val_data.dataset)
        val_loss_list.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(the_model.state_dict(), 'best_model.pt')

        print(f'Epoch {epoch + 1} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}')

    print('Training finished!')


# 定义测试函数
def test(the_model, test_data):
    criterion = nn.MSELoss()
    the_model.load_state_dict(torch.load('best_model.pt'))
    the_model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, Y in tqdm(test_data, desc='Testing'):
            pred_y = the_model(X)
            loss = criterion(pred_y, Y)
            test_loss += loss.item()
        test_loss /= len(test_data.dataset)
    print(f'Test loss: {test_loss:.4f}')


input_size, output_size = dataset.get_data_size()
# 定义模型、训练和测试
model = MyModel(input_size, output_size)
train(model, train_dataloader, val_dataloader, num_epochs=1000, lr=0.001)
test(model, test_dataloader)

fig, ax = plt.subplots()
# 绘制折线图
ax.plot(list(range(100, len(train_loss_list))), train_loss_list[100:], label='train loss')
ax.plot(list(range(100, len(val_loss_list))), val_loss_list[100:], label='val loss')

# 设置图表标题和轴标签
ax.set_title('Train and val loss lines on the model')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')

# 添加图例
ax.legend()

# 显示图表
plt.show()
