

#
try:
    open ("D:/abc.txt","r",encoding="utf-8")
except:
    print("出现异常，因为文件不存在，以open的模式打开")
    open("D:/abc.txt","w",encoding="utf-8")
