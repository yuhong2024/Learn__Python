# 创建一个包
# 导入自定义的包中的模块，并使用
# import my_package.my.module1
# import my_package.my.module2
#
# my_package.my_moddule1.info_print1()
# my_package.my_moddule2.info_print2()

from my_package import my_module1
from my_package import my_module2
my_module1.info_print1()
my_module2.info_print2()

