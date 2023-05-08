import random

# 指定a、b、c的值
import torch

from my_model import MyModel

a = 5
b = 7
c = 9
d = 6
# z = a * x + b * y + c
data = [(random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)) for x, y, z in
        zip(range(1000), range(1000), range(1000))]
the_model = MyModel(3, 1)
the_model.load_state_dict(torch.load('best_model.pt'))
the_model.eval()
for x, y, z in data:
    w = a * x + b * y + c * z + d
    data = torch.tensor([x, y, z], dtype=torch.float32)
    pred_w = the_model(data)
    print(f'TRUE_W={w} | PRED_W={int(pred_w.item())}')
