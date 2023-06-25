from PIL import Image
import os

path = r"C:\Users\DELL\Desktop\1\2"    #图片所在的文件夹路径

for maindir, subdir, file_name_list in os.walk(path):
    for file_name in file_name_list:
        image = os.path.join(maindir, file_name)   #获取每张图片的路径
        file = Image.open(image)
        out = file.resize((1054, 1053), Image.ANTIALIAS)   #以高质量修改图片尺寸为（400，48）
        out.save(os.path.splitext(image)[0] + ".png")   #将保存的格式修改为PNG格式，并保存到原路径下
