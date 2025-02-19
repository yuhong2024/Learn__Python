"""
有一个列表，内容是：[21,25,21,23,22,20],记录的是一批学生的年龄
请通过列表的功能（方法），对其进行
1. 定义这个列表，并用变量接收
2. 追加一个数字31，到列表的尾部
3. 追加一个新列表[29,33,30],到列表的尾部
4. 取出第一个元素（应该是：21）
5. 取出最后一个元素（应该是：30）
6. 查找元素31，在列表中的下标位置
"""


# 1. 定义这个列表，并用变量接收
my_list = [21, 25, 21, 23, 22, 20]
# 2. 追加一个数字31，到列表的尾部
# append
my_list.append(31)
# 3. 追加一个新列表[29,33,30],到列表的尾部
# extend
my_list.extend([29,33,30])
# 4. 取出第一个元素（应该是：21）
num1 = my_list[0]
print(f"从列表中取出的第一个元素：{num1}")

# 5. 取出最后一个元素（应该是：30）
num2 = my_list[-1]
print(f"从列表中取出的第一个元素：{num2}")

# 6. 查找元素31，在列表中的下标位置
index = my_list.index(31)
print(f"元素31在列表中的位置是：{index}")

print(f"最后列表的内容是：{my_list}")

