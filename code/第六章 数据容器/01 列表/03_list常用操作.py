
"""
list列表常用操作
"""
mylist = ["itheima","itcast","python"]

# 1.1 查找某元素在列表内的下标索引
index = mylist.index("python")
print(f"python在列表中的下标索引值是：{index}")


# 1.2 如果被查找的元素不存在，会报错
# index = mylist.index("hi")
# print(f"hi在列表中的下标索引值是：{index}")



# 2 修改特定下标索引的值
mylist[0] = "教育"
print(f"列表修改后，列表内容是{mylist}")


# 3 在指定下标位置插入新元素
my_list = [1,2,3]
print(mylist)
my_list.insert(1,"itheima")
print(f"插入后的列表内容为 ：{my_list}")


# 4. 在列表的尾部追加 单个 新元素
my_list = [1,2,3]
my_list.append(4)
print(my_list) #结果是[1,2,3,4]

# 5. 在列表的尾部追加 一批 新元素
my_list = [1,2,3]
my_list.extend([4,5,6])
print(my_list) #结果是[1，2，3，4，5，6]



# 6. 删除指定下标索引的元素（两种方式
my_list = [1,2,3]
# 6.1 方式1 del 列表[下标]
del my_list[0]
print(my_list) #结果：[2,3]

# 6.2 方式2 列表.pop(下标)
my_list.pop(1)
print(my_list) #结果：[2,3]

# 删除指定元素，remove
my_list = [1,2,3,2,3]
my_list.remove(2)
# 清空列表 clear
my_list.clear()
print(my_list)


# 统计某元素在列表内的数量
# 语法：列表 count(元素)
my_list = [1,1,2,1,3,3]
print(my_list.count(1))

# 统计列表内有多少元素
# 语法：len(列表)
my_list = [1,2,3,4,5,3,4,5,6,1]
print(f"列表的元素数量总共有{len(my_list)}个。")