import time

# 打开文件
f = open("D:/测试.txt","r",encoding="utf-8")
print(type(f))
# 读取文件 -read（）

lines = f.readlines()
print(lines)
print(f"read方法读取全部内容的结果：{f.read()}")


# 读取文件 -readlines()

# 读取文件 -readline()

# for 循环读取文件行

# 文件的关闭

# with open 语法操作文件





f = open("D:/test.txt","w",encoding="utf-8")
f.write("hello world!")
f.flush()
time.sleep(6000000)
