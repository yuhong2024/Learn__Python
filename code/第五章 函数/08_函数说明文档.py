"""
对函数进行文档说明
"""

# 定义函数，说明文档说明
def add(x, y):
    """
    add 函数可以接收2个参数，进行两数相加的功能
    :param x: 形参x表示相加的其中一个数字
    :param y: 形参y白哦是相加的另一个数字
    :return:返回值是两个数字相加的结果
    """
    result = x + y
    print(f"两数相加的结果是：{result}")
    return result

add(1, 2)
add(1, 2)