import xml.etree.ElementTree as ET
from time import sleep
from tqdm import tqdm
from sys import exit
import os


voc_path = './标注/'
out_path = './txt/'


if not os.path.isdir(out_path):
    os.makedirs(out_path)


classes = ["head", "body"]  #类别


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


if __name__ == '__main__':
    if not os.listdir(voc_path):
        exit(0)

    for files in tqdm(os.listdir(voc_path)):
        file_names, file_ext = os.path.splitext(files)
        file_create = open(out_path + file_names + '.txt', 'w+')
        if 'xml' in file_ext:
            tree=ET.parse(voc_path + files)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((w,h), b)
                file_create.write(str(cls_id) + " " + " ".join([str('{:.6f}'.format(a)) for a in bb]) + '\n')

        file_create.close()

    sleep(3)
