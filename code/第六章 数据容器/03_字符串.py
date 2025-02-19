
my_str ="itheima and itcast"
# 通过下标索引取值
value_1 = my_str[2]
vlaue_2 = my_str[3]
print(f"从字符串{my_str}中取下标为2的元素，值是{value_1}，取下标为3的元素，值是{vlaue_2}")

# index方法
value = my_str.index("and")
print(f"在字符串{my_str}中查找and，其起始下标是：{value}")

# replace方法
new_my_str = my_str.replace("and", "and2")
print(new_my_str)

# split 方法
my_str = "itheima and itcast python"
my_str_list = my_str.split(" ")
print(f"将字符串{my_str}进行split切分后得到：{my_str_list}，类型是：{type(my_str_list)}")

