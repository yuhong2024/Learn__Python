"""
contiue语句
"""


# for i in range(1,6):
#     print("语句1")
#     continue
#     print("语句2")

# continue 的嵌套
# for i in range(1,6):
#     print("语句1")
#     for j in range(1,6):
#         print("语句2")
#         continue
#         print("语句3")
#
#     print("语句4")


# break语句
# for i in range(1,10):
#     print("语句1")
#     break
#     print("语句2")
#
# print("语句3")

# break的嵌套使用
for i in range(1,6):
    print("语句1")
    for j in range(1,6):
        print("语句2")
        break
        print("语句3")

    print("语句4")