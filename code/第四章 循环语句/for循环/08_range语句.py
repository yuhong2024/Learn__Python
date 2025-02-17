"""
range语句的使用
"""

#语法1 range((num))
# for x in range(10):
#     print(x)

# 获取一个从0开始，到num结束的数字序列，但不包括num本身
# 如range(5)获得的数据是：{0，1，2，3，4}

# 语法2 range(num1,num2)
# for x in range(6,11):
#     print(x)
# 获得一个从num1开始，到num2结束的数字序列，但不含num2本身
#如range(6,11)取得的数据是{6，7，8，9，10}但是不包括11

#语法3 range(num1,num2,step)
for x in range(3,10,2):
    print(x)

#获得一个从num1开始，到num2结束的数字序列，这段数字之间的步长，以step为准
#如range{5,10,2}取得的数据是{5，7，9}


