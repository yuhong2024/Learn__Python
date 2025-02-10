# 将input输入语句直接写入判断条件中
print("欢迎来到动物园")
if int(input("请输入您的身高(cm):")) < 120 :
    print("您的身高小于120cm，可以免费游玩。")
elif int(input("请输入您的vip级别（1~50）：")):
    print("您的vip级别大于3，可以免费游玩。")
elif int (input("请输入今天的提起（1~30）")):
    print("今天是1号免费日，可以免费游玩。")
else:
    print("不好意思，所有条件都不满足，需要购票10元。")

print("祝您游玩愉快。")




print("欢迎来到黑马动物园")
height = int(input("请输入您的身高（cm):"))
vip_level = int(input("请输入您的vip级别（1~5）："))
day = int (input("请输入今天的日期(1~30)："))

if height < 120 :
    print("您的身高小于120cm，可以买免费游玩。")
elif vip_level > 3:
    print("您的vip等级大于3，可以免费游玩。")
elif day == 1:
    print("今天是1号免费日，可以免费游玩")
else :
    print("不好意思，所有条件都不满足，需要购票10元。")

print("祝您游玩愉快。")


