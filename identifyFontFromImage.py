import halcon as ha
import numpy as np
import re
import time
import os
from textVision import get_words_boxes
from utils import charPath, draw_boxes, drawScores

FONTLIST = []
FONTLISTLEN = 0

def identifyFontFromImage(imagePath: str, maxCharMatchNum: int = 5, useGrayImage: bool = True):
    """
    从给定的候选字体列表fontList中推测图片中的字体

    Input: \\
    imagePath(str): 单行文本图片路径；\\
    maxCharMatchNum(int): 用来进行模板匹配的字符最大数量；\\
    useGrayImage(bool): 是否使用灰度图来进行模板匹配。

    Output: \\
    predictFont(str): 推断出的字体类型；\\
    predictScores(list): 每种字体的概率分布。
    """
    # 耗时统计
    start_time = None
    end_time = None
    # 输出
    predictFont = ''
    predictScores = None
    # 中间变量
    predictFontScores = []

    # GOOLE VISION识别图片中的字符，估计大小
    start_time = time.time()
    boxes, estFontsize = get_words_boxes(imagePath, bleeding=0.2)
    end_time = time.time()
    print(f"Google Vision：{end_time - start_time:.3f} 秒")

    # TODO:根据特征字符集选出charList
    charList = []
    for i in range(min(len(boxes), maxCharMatchNum)):
        char = boxes[i]['text']
        if re.match(r'[A-Za-z0-9]', char) and char!='I' and char!='l':
            charList.append(boxes[i])

    # 根据estFontsize确定模板尺寸
    if estFontsize < 50:
        templateSize = '50'
    elif estFontsize < 100:
        templateSize = '100'
    elif estFontsize < 200:
        templateSize = '200'
    # elif estFontsize < 400:
    #     templateSize = '400'
    
    # 读取目标图片
    image = ha.read_image(imagePath)
    # # 获取图片高度和宽度
    # imageWidth, imageHeight = ha.get_image_size(image)
    # 转为灰度图
    grayImage = ha.rgb1_to_gray(image)

    # 对每个挑选出的字符进行模板匹配
    start_time = time.time()
    for char in charList:
        scores = [0] * FONTLISTLEN
        # TODO: 思考如何减少遍历字体次数，启发式算法，而不是傻瓜式
        for i in range(FONTLISTLEN):
            # 读取模板
            ModelID = ha.read_shape_model('templates/{}/{}/{}.shm'.format(FONTLIST[i], templateSize, charPath(char['text'])))
            # 模板匹配参数设置
            ha.set_generic_shape_model_param(ModelID, 'num_matches', 1)
            ha.set_generic_shape_model_param(ModelID, 'min_score', 0.3)
            ha.set_generic_shape_model_param(ModelID, 'border_shape_models', 'false')
            # 模板匹配
            ModelRegion = ha.gen_rectangle1(char['xyxy'][1], char['xyxy'][0], char['xyxy'][3], char['xyxy'][2])
            croppedCharImage = ha.reduce_domain(grayImage, ModelRegion)
            MatchResultID, NumMatchResult = ha.find_generic_shape_model(croppedCharImage, ModelID)
            if NumMatchResult > 0:
                scores[i] = ha.get_generic_shape_model_result(MatchResultID, 0, 'score')[0]
        end_time = time.time()
        # print("{}:{}".format(char['text'], scores))
        predictFontScores.append(scores)
    end_time = time.time()
    print(f"匹配识别：{end_time - start_time:.3f} 秒")
    predictFontScores = np.average(predictFontScores, axis=0)
    predictScores = {k: v for k, v in zip(FONTLIST, predictFontScores)}
    matchedFontIndex = np.argsort(-np.array(predictFontScores))[0]
    predictFont = FONTLIST[matchedFontIndex]

    return predictFont, predictScores

if __name__ == '__main__':
    for file in os.listdir('fonts'):
        fontName,_ = os.path.splitext(file)
        FONTLIST.append(fontName)

    FONTLISTLEN = len(FONTLIST)

    test_font = 'OPENSANS-MEDIUM'
    imagePath = 'imgs/{}.jpg'.format(test_font)
    predictFont, predictScores = identifyFontFromImage(imagePath, maxCharMatchNum=10)
    # print(predictFont, predictScores)
    drawScores(test_font, predictScores)
