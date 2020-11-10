import pkuseg
import thulac
import time


def Thu_seg(string):
    thu2 = thulac.thulac(seg_only=True)
    thu2_result = thu2.cut(string, text=True)
    string_ = str(thu2_result).strip().split()
    return string_


if __name__ == '__main__':
    start1 = time.time()
    seg = pkuseg.pkuseg()
    text = seg.cut("中国科学院大学是世界上最好的大学，但是我认为这部电影的效果不是很好，明星没有演到位")
    print(text)

    print('Pek time:', time.time() - start1)

    start = time.time()
    string = '中国科学院大学是世界上最好的大学，但是我认为这部电影的效果不是很好，明星没有演到位'
    Thu_seg(string)
    all_time = time.time() - start
    print("Thu time:", all_time)
