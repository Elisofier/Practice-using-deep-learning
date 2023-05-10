import math

# 生成sin函数的值
x_values = [i / 10 for i in range(0, 100)]
y_values = [math.sin(x) for x in x_values]

# 将结果写入文件
with open('data.txt', 'w') as f:
    for y in y_values:
        f.write(f'{y}\n')
# 生成sin函数的值
x_values = [i / 10 for i in range(0, 100)]
y_values = [math.cos(x) for x in x_values]

# 将结果写入文件
with open('test.txt', 'w') as f:
    for y in y_values:
        f.write(f'{y}\n')
