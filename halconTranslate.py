"""
VERSION1，即将废弃，仅作参考
"""
import halcon as ha
import numpy as np
import matplotlib.pyplot as plt
import time
import math

def getFontFromImage(imagePath: str, fontList: list, isBold: bool, isSlant: bool):
    """
    从给定的候选字体列表fontList中推测图片中的字体

    Input: \\
    image(str): 单行文本图片路径；\\
    charList(list): 图片中出现的字符列表；\\
    estFontsize(int): 字符估计大小；\\
    fontList(list): 候选字体列表；\\
    isBold(bool): 字符是否有加粗；\\
    isSlant(bool): 字符是否有倾斜。

    Output: \\
    predictFont(str): 推断出的字体类型；\\
    predictScores(list): 每种字体的概率分布。
    """
    start_time = None
    end_time = None
    # start_time2 =  None
    # end_time2 = None
    # 输出
    predictFont = ''
    predictScores = None
    # 准备工作
    fontListLen = len(fontList)
    fontStyle = ''
    if isBold:
        fontStyle = fontStyle + 'Bold'
    if isSlant:
        fontStyle = fontStyle + 'Italic'
    if fontStyle == '':
        fontStyle = 'Normal'
    predictFontsScore = []
    # predictFontHeights = {key: None for key in charList}
    # 创建绘制窗口
    WindowID = ha.open_window(0,0,900,900,0,'visible','')
    # 使用预训练的OCR
    TextModel = ha.create_text_model_reader('auto', 'Document_NoRej')
    for char in charList:
        scores = [0] * fontListLen
        # 遍历所有候选字体，分别制作模板进行匹配，记录匹配分数
        for i in range(fontListLen):
            start_time = time.time()
            # 设置字体样式并绘制于窗口
            ha.set_font(WindowID, fontList[i] + '-' + fontStyle + '-' + str(300))
            ha.write_string(WindowID, char+' '+char)
            # ha.wait_seconds(2)
            textImage = ha.dump_window_image(WindowID)
            # 使用OCR识别文字位置并获得最小包裹矩形框
            TextResultID = ha.find_text(textImage, TextModel)
            Characters = ha.get_text_object(TextResultID, 'all_lines')
            SingleChar = ha.select_obj(Characters, 1)
            if ha.count_obj(SingleChar) == 1:
                Row1, Column1, Row2, Column2 = ha.smallest_rectangle1(SingleChar)
                # 计算fontHeight
                # predictFontHeights[char] = Row2[0] - Row1[0]
                # 制作模板
                ModelRegion = ha.gen_rectangle1(Row1[0], Column1[0], Row2[0], Column2[0])
                TemplateImage = ha.reduce_domain(textImage, ModelRegion)
                ModelID = ha.create_scaled_shape_model(TemplateImage, 5, math.radians(0), math.radians(0), math.radians(0), estFontsize * 0.8 / 100, estFontsize * 1.2 / 100, 'auto', 'none', 'ignore_global_polarity', 40, 10)
                end_time2 = time.time()
                ha.clear_window(WindowID)
                # 模板匹配
                ha.set_generic_shape_model_param(ModelID, 'num_matches', 1)
                ha.set_generic_shape_model_param(ModelID, 'min_score', 0.3)
                ha.set_generic_shape_model_param(ModelID, 'border_shape_models', 'false')
                image = ha.read_image(imagePath)
                MatchResultID, NumMatchResult = ha.find_generic_shape_model(image, ModelID)
                if NumMatchResult > 0:
                    scores[i] = ha.get_generic_shape_model_result(MatchResultID, 0, 'score')[0]
            end_time = time.time()
            print(f"一次运行耗时：{end_time - start_time} 秒")
            # print(f"制作模板占比耗时：{(end_time2 - start_time) / (end_time - start_time)}")
            ha.clear_window(WindowID)
        predictFontsScore.append(scores)

    predictFontsSumScore = np.sum(predictFontsScore, axis=0)
    sumScore = np.sum(predictFontsSumScore)
    predictScores = [score / sumScore for score in predictFontsSumScore]
    predictScores = {k: v for k, v in zip(fontList, predictScores)}
    matchedFontIndex = np.argsort(-np.array(predictFontsSumScore))[0]
    predictFont = fontList[matchedFontIndex]
    ha.close_window(WindowID)

    return predictFont, predictScores

def drawScores(correctFont:str, scores:dict):
    # 获取前5个概率最高的字体及其对应的概率
    sorted_data = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    labels, probabilities = zip(*sorted_data)

    # 绘制柱形图
    colors = ['#ffaaaa' if label == correctFont else '#aaaaff' for label in labels]
    plt.barh(labels, probabilities, color=colors, edgecolor='black')

    # 反转 y 轴坐标
    plt.gca().invert_yaxis()

    # 添加标题和标签
    plt.title('Top 5 Font Probabilities')
    plt.xlabel('Probability')

    # 显示图形
    plt.show()

if __name__ == "__main__":
    fontLabel = 'Roboto'
    fontList = ['Poppins', 'Roboto', 'Abel', 'Arima', 'Halant', 'Open Sans', 'Alef', 'Ubuntu', 'Saira', 'Lato']
    isBold = False
    isSlant = False
    start_time = time.time()
    font, scores = getFontFromImage(imagePath, charList, estFontsize, fontList, isBold, isSlant)
    end_time = time.time()

    print(f"程序运行耗时：{end_time - start_time} 秒")
    print(scores)
    drawScores(fontLabel, scores)
