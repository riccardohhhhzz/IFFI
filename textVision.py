import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.oauth2 import service_account

# Instantiates a client
credentials = service_account.Credentials.from_service_account_file(
    'doraoauth-1e8e4205584a.json')
client = vision.ImageAnnotatorClient(credentials=credentials)

def getXY(xy):
    string = str(xy)
    if "x" in string:
        x_index = string.index("x:") + 2
        x_value = int(string[x_index:string.index("\n", x_index)])
    else:
        x_value = 0

    if "y" in string:
        y_index = string.index("y:") + 2
        y_value = int(string[y_index:])
    else:
        y_value = 0

    return [x_value, y_value]

def get_words_boxes(path, bleeding=0.2):    
    """返回word的左上和右下顶点[x,y,x,y]，识别出的字符，以及得分"""
    estFontsize = 0
    with open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    # 获取最顶层page
    response_page = response.full_text_annotation.pages[0]
    # 进行blocks->paragraphs->words遍历得到所有的words
    words = []
    for b in response_page.blocks:
        for p in b.paragraphs:
            for w in p.words:
                words.extend(w.symbols)
    output = []
    for w in words:
        xyxy = []
        xyxy.extend(getXY(w.bounding_box.vertices[0]))
        xyxy.extend(getXY(w.bounding_box.vertices[2]))
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        estFontsize += height * (1+bleeding*2)
        xyxy = [int(xyxy[0] - width*bleeding), int(xyxy[1] - height*bleeding), int(xyxy[2] + width*bleeding), int(xyxy[3] + height*bleeding)]
        output.append({
            "xyxy": xyxy,
            "text": w.text
        })
    estFontsize = int(estFontsize / len(output))
    return output, estFontsize

if __name__ == "__main__":
    # run_quickstart()
    # detect_text('imgs/118_1.jpg')
    boxes = get_words_boxes('imgs/Lato_test.jpg')
    print(boxes)
