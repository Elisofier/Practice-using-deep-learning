import random

# 指定a、b、c的值
a = 5
b = 7
c = 9
d = 6
# z = ax + by + cz + d
data = [(random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)) for x, y, z in
        zip(range(1000), range(1000), range(1000))]

with open("data.txt", "w") as f:
    for x, y, z in data:
        f.write(f"{x}\t{y}\t{z}\n")
with open("label.txt", "w") as f:
    for x, y, z in data:
        f.write(f"{a * x + b * y + c * z + d}\n")
