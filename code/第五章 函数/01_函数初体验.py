"""
体验函数的开发与应用
"""

# 需求：统计字符串的长度，不适用内置函数

str1 = "ithema"
str2 = "itcastone"
str3 = "pythonlearn"

# 定义一个计数的变量
count = 0
for i in str1:
    count += 1
print(f"字符串{str1}的长度是：{count}")

count = 0
for i in str2:
    count += 1
print(f"字符串{str2}的长度是：{count}")

count = 0
for i in str3:
    count += 1
print(f"字符串{str3}的长度是：{count}")


# 使用函数，优化这个过程
def my_len(data):
    count = 0
    for i in data:
        count += 1
    print(f"字符串{data}的长度是：{count}")

my_len(str1)
my_len(str2)
my_len(str3)
