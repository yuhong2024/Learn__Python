"""
list列表
"""

# 定义一个列表 list
["itheima",1211212,12.10,"python"] # 字面量
my_list = [1,2,3,4,5] #变量
print(my_list)
print(type(my_list))

# 定义一个嵌套的列表
my_list = [[1,2,3],[3,2,3],[5,6,7]]
print(my_list)
print(type(my_list))

# 通过下标索引取出对应位置的数据
my_list = ["Tony","Jack","Rose"]
# 列表[下标索引] 从前向后从0开始，每次+1，从后向前从-1开始，每次-1
print(my_list[0])
print(my_list[1])
print(my_list[2])
# 错误示范：通过下标索引取数据，颐堤港不要超出范围
# print("my_list[3]")

# 通过下标索引去除数据（倒序取出）
print(my_list[-1])
print(my_list[-2])
print(my_list[-3])

# 取出嵌套列表的元素
my_list = [[1,2,3],[3,2,3],[5,6,7]]
print(my_list[2][0])