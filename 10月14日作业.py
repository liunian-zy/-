import jieba.posseg as pseg
import jieba
import jieba.analyse


# 关键词提取
def fun1():
    # 导入自定义词典
    jieba.load_userdict("dict.txt")
    # 精确模式
    text = "故宫的著名景点包括乾清宫、太和殿和午门等。其中乾清宫非常精美，午门是紫禁城的正门，午门剧中向阳。"
    seg_list = jieba.cut(text, cut_all=False)
    print(u"分词结果:")
    print("/".join(seg_list))

    # 获取关键词
    tags = jieba.analyse.extract_tags(text, topK=3)
    print(u"关键词:")
    print(" ".join(tags))


# 词性标注
def fun2():
    words = pseg.cut("我爱北京天安门，天安门上太阳升！")
    for w in words:
        print('%s %s' % (w.word, w.flag))


# 结巴分词
def fun3():
    seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
    print("Full Mode: " + "/".join(seg_list))
    seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
    print("Default Mode: " + "/".join(seg_list))
    seg_list = jieba.cut("他来到了网易行研大厦")
    print(", ".join(seg_list))
    seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
    print(", ".join(seg_list))


# 添加用户词典1
def fun4():
    # 导入自定义词典
    jieba.load_userdict("dict.txt")
    # 全模式
    text = "故宫的著名景点包括乾清宫、太和殿和午门等。其中乾清宫非常精美，午门是紫禁城的正门，午门剧中向阳。"
    seg_list = jieba.cut(text, cut_all=True)
    print(u"[全模式]: " + "/".join(seg_list))
    seg_list = jieba.cut(text, cut_all=False)
    print(u"[精确模式]: " + "/".join(seg_list))
    seg_list = jieba.cut_for_search(text)
    print(u"[搜索引擎模式]: " + "/".join(seg_list))


# 添加用户词典2
def fun5():
    jieba.load_userdict("userdict.txt")
    jieba.add_word("石墨烯")
    jieba.add_word("凯特琳")
    jieba.del_word("自定义词")
    test_send = (
        "李小福是创新办主任也是云计算方面的专家；什么是八一双鹿\n"
        "例如我输入一个带“韩玉鉴赏”的标题，在自定义词库中也增加了此词为N类\n"
        "「台中」正確應該不會被切開。mac上可以分出「石墨烯」；此時又可以分出來凱特琳了。"
    )
    words = jieba.cut(test_send)
    print("/".join(words))
    print("=" * 40)
    result = pseg.cut(test_send)
    for w in result:
        print(w.word, "/", w.flag, ", ", end="  ")
    print("\n" + "=" * 48)

    terms = jieba.cut("easy_install is great")
    print("/".join(terms))
    terms = jieba.cut("python 的正则表达式是好用的")
    print("/".join(terms))

    print("=" * 40)

    testlist = [
        ('今天天气不错', ('今天', '天气')),
        ('如果放到post中将出错。', ('中', '将')),
        ('我们中出了一个叛徒', ('中', '出'))
    ]
    for send, seg in testlist:
        print("/".join(jieba.cut(send, HMM=False)))
        word = ''.join(seg)
        print("%s Before: %s, After:  %s" % (word, jieba.get_FREQ(word), jieba.suggest_freq(seg, True)))
        print("/".join(jieba.cut(send, HMM=False)))
        print("-" * 40)


if __name__ == '__main__':
    fun1()
    fun2()
    fun3()
    fun4()
    fun5()
