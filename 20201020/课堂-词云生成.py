from pyecharts import options as opts  # 绘制图画大小样式等
from pyecharts.charts import WordCloud  # 词云库
from pyecharts.globals import SymbolType

from collections import Counter
import time
import jieba

import imageio
import numpy as np

# mk = np.array(imageio.imread("111.png"))
mk = "1212.jpeg"
# print(mk)
# 导入自定义词典
# jieba.load_userdict("userdict.txt")

cut_words = ""
all_words = ""
f = open('词云原始文本分词1.txt', 'w', encoding='utf-8')  # 创建新的txt
for line in open('词云原始文本1.txt', encoding='utf-8'):
    line.strip('\n')  # 去除首尾空行
    seg_list = jieba.cut(line, cut_all=False)  # 设默认值为精确模式，
    # 返回的是可迭代的generator,cut_all判断是否为全模式
    # print(" ".join(seg_list))#join将序列中的元素以指定字符串连接成\
    # 一个新的字符串
    cut_words = (" ".join(seg_list))
    f.write(cut_words)
    all_words += cut_words
else:
    f.close()
# 输出结果
# print(all_words)
all_words = all_words.split()  # 将字符串分割
# print(all_words)
# 词频统计
c = Counter()  # 统计每个词出现的次数
for x in all_words:
    print(x)
    if len(x) > 1 and x != '\r\n':
        c[x] += 1

# 输出词频最高的前10个词
print('\n词频统计结果：')
# print(c.most_common())
# most_common返回的是TopN列表·
# print(c)
for (k, v) in c.most_common(10):
    print("%s:%d" % (k, v))
# 存储数据
name = time.strftime("%Y-%m-%d") + "-fc.csv"  # 命名包含当前时间的csv文件
fw = open(name, 'w', encoding='utf-8')
i = 1
for (k, v) in c.most_common(len(c)):
    fw.write(str(i) + ',' + str(k) + ',' + str(v) + '\n')
    i = i + 1
else:
    print("Over write file")
    fw.close()

# 词云分析
words = []
# 将词频统计结果加入words列表中
for (k, v) in c.most_common(100):
    words.append((k, v))


# 渲染图
def word_cloud_base() -> WordCloud:
    return WordCloud().add("", words, word_size_range=[10, 50],
                           shape=SymbolType.ROUND_RECT,
                           mask_image=mk,
                           # width="500",
                           height="750",
                           pos_top="20",
                           emphasis_shadow_color='red',
                           rotate_step=135).set_global_opts(title_opts=opts.TitleOpts(title='报道词云图'))


word_cloud_base().render('词云图.html')  # 用Chrome打开

# 核心代码为：
# add(name, attr, value, shape=“circle”, word_gap=20, word_size_range=None,\
# rotate_step=45)
#
# name -> str: 图例名称
# attr -> list: 属性名称
# value -> list: 属性所对应的值
# shape -> list: 词云图轮廓，有’circle’, ‘cardioid’, ‘diamond’, \
# ‘triangleforward’, ‘triangle’, ‘pentagon’, ‘star’可选
# word_gap -> int: 单词间隔,默认为20
# word_size_range -> list: 单词字体大小范围,默认为[12,60]
# rotate_step -> int: 旋转单词角度,默认为45
