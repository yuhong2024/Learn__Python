"""
遍历字符串 "itheima is a brand of itcast"统计有多少个英文字母a
"""

# 定义字符串name
name = "itheima is a brand of itcast"

# 定义一个变量，来统计有多少个a
count = 0

# for 循环统计
# for 临时变量 in 被统计的数据：
for x in name:
    if x == "a":
        count += 1

print(f"被统计的字符串中有{count}个a")

