
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

# 5. 在列表的尾部追加 一批 新元素

# 6. 删除指定下标索引的元素（两种方式

# 6.1 方式1 del 列表[下标]

# 6.2 方式2 列表.pop(下标)