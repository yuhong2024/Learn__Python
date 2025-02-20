

# 基础折线图
# 导包，导入line功能构建折线图对象
from pyecharts.charts import Line
from pyecharts.options import TitleOpts,LegendOpts,ToolboxOpts,VisualMapOpts
# 得到折线图对象
line = Line()
# 添加x轴数据
line.add_xaxis(["中国","美国","英国"])
# 添加y轴数据
line.add_yaxis("GDP",[30,20,10])
# 设置全局配置项set_global_opts来设置
line.set_global_opts(
    title_opts=TitleOpts(title= "GDP展示",pos_left="center",pos_bottom="1%"),
    legend_opts=LegendOpts(is_show=True),
    toolbox_opts=ToolboxOpts(is_show=True),
    visualmap_opts=VisualMapOpts(is_show=True),


)


# 生成图表
line.render()


