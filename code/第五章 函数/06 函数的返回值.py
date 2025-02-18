# 定义一个函数，完成两数相加
def add(a,b):
    result = a + b
    # 通过返回值，将相加的结果返回给调用者
    return result
    # 返回结果后，还想输出一句话
    # print("我结束了")  不执行

# 函数的返回值，通过变量接收
r = add(5, 6)
print(r)

