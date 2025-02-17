"""
while循环 猜数字
"""

# 获取0-100的随机数字
import random

from numpy.testing.print_coercion_tables import print_new_cast_table

num = random.randint(1, 100)
# print(num)
# 定义一个变量，记录总共猜测多少次
count = 0


# 通过一个布尔类型的变量，做循环是否继续标记
flag = True
while flag:
    guee_num = int(input("请输入你猜测的数字："))
    count += 1
    if guee_num == num:
        print("猜中了")
        # 设置False就是终止循环的条件
        flag = False
    else :
        if guee_num > num:
            print("你猜的大了")
        else :
            print("你猜的小了")

print(f"你总共猜测了{count}次")