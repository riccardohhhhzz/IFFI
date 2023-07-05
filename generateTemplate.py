import halcon as ha
from PIL import Image, ImageDraw, ImageFont
import os
import math
from utils import charPath

def generateTemplate(fontname:str, fontsize:int=100):
    if os.path.exists('templates_raw/{}/{}'.format(fontname, fontsize)):
        return
    if os.path.exists('templates/{}/{}'.format(fontname, fontsize)):
        return
    
    # 字体文件路径
    font_path = 'fonts/{}.ttf'.format(fontname)

    # 创建空白图像
    image_width = 4000
    image_height = 2400
    image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 字体大小和起始位置以及间隙
    x_start = 30
    y_start = 30
    x_gap = 30
    y_gap = 30
    bleeding = 2

    # 加载字体
    font = ImageFont.truetype(font_path, fontsize)

    # 绘制字符(不要I和l)
    characters = 'ABCDEFGHJKLMNOPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz0123456789'
    x = x_start
    y = y_start

    boxes = []

    for char in characters:
        # 获取字符的宽度和高度
        width, height = draw.textsize(char, font=font)
        
        # 绘制字符
        draw.text((x, y), char, font=font, fill=(0, 0, 0))

        # 获取绘制字符的最小矩形框
        bbox = draw.textbbox((x, y), char, font=font)
        boxes.append({
            'xyxy': [bbox[0] - bleeding, bbox[1] - bleeding, bbox[2] + bleeding, bbox[3] + bleeding],
            'text': char
        })

        # 更新下一个字符的起始位置
        x += width + x_gap
        
        # 换行
        if x >= image_width - width - x_gap:
            x = x_start
            y += height + y_gap

    # Debug
    # draw_boxes(boxes, np.asarray(image))

    # 保存图像为JPEG
    if not os.path.exists('templates_raw/{}'.format(fontname)):
        os.makedirs('templates_raw/{}'.format(fontname))
    os.makedirs('templates_raw/{}/{}'.format(fontname, fontsize))

    for box in boxes:
        cropped_image = image.crop(box['xyxy'])
        cropped_image.save('templates_raw/{}/{}/{}.jpg'.format(fontname, fontsize, charPath(box['text'])))
    # image.save(template_raw_path)

    # 制作模板
    if not os.path.exists('templates/{}'.format(fontname)):
        os.makedirs('templates/{}'.format(fontname))
    os.makedirs('templates/{}/{}'.format(fontname, fontsize))

    for char in characters:
        CharImage = ha.read_image('templates_raw/{}/{}/{}.jpg'.format(fontname, fontsize, charPath(char)))
        ModelID = ha.create_scaled_shape_model(CharImage, 5, math.radians(0), math.radians(0), math.radians(0), 0.2, 1.1, 'auto', 'none', 'ignore_global_polarity', 40, 10)
        ha.write_shape_model(ModelID, 'templates/{}/{}/{}.shm'.format(fontname, fontsize, charPath(char)))

if __name__ == '__main__':
    fontlist = []
    for file in os.listdir('fonts'):
        fontName,_ = os.path.splitext(file)
        fontlist.append(fontName)
    templateSize = [50, 100, 200]

    for font in fontlist:
        for s in templateSize:
            generateTemplate(font, s)
