"""
变量的作用域
"""

# 局部变量
# def testA():
#     num = 100
#     print(num)
# testA()
# print(num) #报错，变量a是定义在testA韩式内部的变量，在函数外部访问则立即报错

# 定义全局变量
# num = 100
# def testA():
#     print(num) # 访问全局变量num,并打印变量num存储的数据
#
# def testB():
#     print(num) # 访问全局变量num，并打印变量num存储的数据
#
# testA() # 100
# testB() # 100
#
# print(num)

# global关键字
num = 200

def test_a():
    print(f"test_a:{num}")

def test_b():
    global num
    num = 500
    print(f"test_b:{num}")

test_a()
test_b()
print(num)