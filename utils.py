import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def charPath(char:str):
    if char.isupper():
        return char.lower()+'_b'
    else:
        return char
    
def draw_boxes(boxes, image, random_color=True):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    image = Image.fromarray(image)
    # 创建draw对象
    draw = ImageDraw.Draw(image)
    for box in boxes:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        color_tuple = tuple((color*255).astype(np.uint8))
        # 绘制边框
        draw.rectangle(box['xyxy'], outline=color_tuple, width=1)
        # 绘制标签
        label_xyxy = [box['xyxy'][0], box['xyxy'][1]-8, box['xyxy'][0] + 12, box['xyxy'][1]]
        draw.rectangle(label_xyxy, fill=color_tuple)
        # 获取文本宽度和高度
        if len(box['text']) == 1:
            box['text'] = box['text'].encode('utf-8')
            text_width, text_height = draw.textsize(box['text'])
            text_x = (label_xyxy[0] + label_xyxy[2] - text_width) // 2
            text_y = (label_xyxy[1] + label_xyxy[3] - text_height) // 2
            text_color = (255,255,255)
            draw.text((text_x, text_y), box['text'], fill=text_color, anchor="mm")
    image.show()
    return np.array(image)

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

    # 在每个柱形内部添加概率值
    for i in range(len(probabilities)):
        plt.text(probabilities[i], i, f"{probabilities[i]:.3f}", ha='right', va='center')
    
    plt.subplots_adjust(left=0.28, right=0.9, bottom=0.1, top=0.9)
    # 显示图形
    plt.show()
