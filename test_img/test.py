import os
import extcolors
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image
import matplotlib.colors as mcolors
import pandas as pd

from ultralytics import YOLO

import torch
from torchvision.transforms import v2 # pytorch

# 함수 #####################

def crop_imgs(img_path, custom_model):
    model = YOLO('yolov8m.pt')  # load a pretrained YOLOv8 detection model
    model.train(data='C:/Users/Andlab/jupyter_study/DL_project/Clothes-Category-9/data.yaml', epochs=500, imgsz=640)  # train the model
    # 혹은 학습한 모델을 model = YOLO() 안에 넣어줘도 됨
    model.predict(img_path, save_crop=True)  # predict on an image 

# XKCD COLORS info를 df로 생성
def colors_df():
  global df
  df = pd.DataFrame(columns=['name','red','green','blue'])
  for color_name, color_hex in mcolors.XKCD_COLORS.items():
      r, g, b = mcolors.to_rgb(color_hex)
      df = pd.concat([df, pd.DataFrame({'name':[color_name], 'red':[r], 'green':[g], 'blue':[b]})], ignore_index= True)
  return df

# 가까운 색상명
def closest_color(rgb):
  differences = {}
  for color_name, color_hex in mcolors.XKCD_COLORS.items(): # mcolors.CSS4_COLORS mcolors.XKCD_COLORS
    r, g, b = mcolors.to_rgb(color_hex)

    differences[sum([(r*255-rgb[0])**2,
                    (g*255-rgb[1])**2,
                    (b*255-rgb[2])**2])] = color_name
    
  return differences[min(differences.keys())]


# 이미지 색상명 추출
def find_colorname(item, croppedfolder):
  top_img = os.listdir(croppedfolder + item)[0]
  top_img_path = croppedfolder + item + '/' + top_img

  org_img = Image.open(top_img_path)

  # 원본 이미지 사이즈 구하기
  org_img_size = v2.functional.get_size(org_img)

  # 가로, 세로 중 짧은 쪽의 80% 길이로 만들기
  size_80 = min(org_img_size[0], org_img_size[1]) * 0.8
  centercrop = v2.CenterCrop(size_80)
  img_size80 = centercrop(org_img)

  # 색상 추출
  colors, pixel_count = extcolors.extract_from_image(img_size80)

  # output
  clothes_color = closest_color(colors[0][0])

  return clothes_color

# 파일명 없으면 생성
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# run ###############################################

# 색상, 옷 카테고리 결과 담을 df
result = pd.DataFrame(columns=['img_path','top_category','top_color','bottom_category','bottom_color'])

# source info
image_folder = '.\\images\\'
custom_model = '..\\object_detection\\runs\\detect\\datav5result\\weights\\best.pt'

for img in os.listdir(image_folder):
    image_source = image_folder + img

    try:
        # object에 맞게 이미지 크롭
        crop_imgs(image_source, custom_model)

        # 크롭된 이미지의 위치
        cropped_folder = r'./runs/detect/predict/crops/'
        for item in os.listdir(cropped_folder): # 해당 폴더 내 모든 파일 및 폴더 추출
            # item: 옷 카테고리명
            color = find_colorname(item, cropped_folder)
            if item in ['shirt','sweater','mtm','hoodies','longsleeve','shortsleeve','jacket','blazer','padding','zipup','cardigan','coat']:
                top = {'item':item, 'color':color}
            else: # skirt, denim pants, cotton pants, trainingpants, slacks, short pants
                bottom = {'item':item, 'color':color}

        # result df에 추가
        result = pd.concat([result, pd.DataFrame({'img_path':[image_source],'top_category':[top.get('item')],
                                                'top_color':[top.get('color')],'bottom_category':[bottom.get('item')],
                                                'bottom_color':[bottom.get('color')]})], ignore_index= True)
            
        # cropped_folder 삭제
        import shutil
        shutil.rmtree(r'./runs/')
    except:
        continue

result.to_csv('./result.csv',index=False, encoding='utf-8')


# color #################################
color_df = colors_df()

green = ['green','apple', 'algae', 'asparagus', 'avocado', 'poo', 'poop', 'bile', 'booger', 'lime', 'olive', 'khaki']

black = ['black','charcoal','dark']

blue = ['aqua', 'blue', 'marine', 'azul', 'azure', 'blueberry', 'bluish', 'cyan', 'teal', 'cobalt', 'cornflower', 'navy']

grey = ['grey','steel','silver','cement','greyish','gunmetal']

white = ['white']

orange = ['orange']

red = ['red','blood','berry','rose']

purple = ['amethyst','purple', 'barney', 'violet', 'lavender', 'bruise', 'lilac', 'eggplant','indigo','iris']

pink = ['pink', 'blush', 'magenta', 'bubblegum']

brown = ['brown', 'bronze', 'brownish', 'brick', 'burgundy', 'sienna', 'umber', 'chocolate', 'chestnut', 'cinnamon', 'clay', 'coffee', 'copper']

yellow = ['yellow', 'banana', 'buff', 'butter', 'cream', 'creme', 'custard', 'dandelion']

color_df['main_color'] = ''

for i, c in enumerate(color_df['name']):

    color = c.replace('xkcd:','').split(' ')[-1]

    if color in green:
        color_df['main_color'][i] = 'green'
    elif color in black:
        color_df['main_color'][i] = 'black'
    elif color in blue:
        color_df['main_color'][i] = 'blue'
    elif color in grey:
        color_df['main_color'][i] = 'grey'
    elif color in white:
        color_df['main_color'][i] = 'white'
    elif color in orange:
        color_df['main_color'][i] = 'orange'
    elif color in red:
        color_df['main_color'][i] = 'red'
    elif color in purple:
        color_df['main_color'][i] = 'purple'
    elif color in pink:
        color_df['main_color'][i] = 'pink'
    elif color in brown:
        color_df['main_color'][i] = 'brown'
    elif color in yellow:
        color_df['main_color'][i] = 'yellow'
    else:
        color_df['main_color'][i] = 'others'

color_df.to_csv('./maincolors.csv',encoding='utf-8',index=False)

# concat dataset ########################
maincolors= pd.read_csv('./maincolors.csv').drop(columns=['red','blue','green'])
result = pd.read_csv('./result.csv')

result = pd.merge(result, maincolors, how='left', left_on='top_color', right_on='name').drop(columns='name').rename(columns={'main_color':'top_color_cluster'})
result = pd.merge(result, maincolors, how='left', left_on='bottom_color', right_on='name').drop(columns='name').rename(columns={'main_color':'bottom_color_cluster'})

result.to_csv('./result_maincolors.csv',encoding='utf-8',index=False)


# visualization #########################
from PIL import Image
import matplotlib.pyplot as plt

# // %matplotlib inline

plt.rcParams['figure.figsize'] = (15.0, 130.0)



# struct is  [10, 5]
rows = 20
columns = 5

for i in range(len(result[:50])) : 
    image = Image.open(result['img_path'][i])
    image_index = i + 1     # image index 
    ttitle = f"top: {result['top_color'][i]}({result['top_color_cluster'][i]})\n{result['top_category'][i]}" # image title
    plt.subplot(rows, columns, image_index) # subplot 
    plt.title(ttitle, fontsize=10)   # title 
    # // plt.axis('off')
    plt.xticks([])  # x = None 
    plt.yticks([])  # y = None
    plt.xlabel(f"bottom: {result['bottom_color'][i]}({result['bottom_color_cluster'][i]})\n{result['bottom_category'][i]}", fontsize=10)
    plt.imshow(image)

plt.show()
#############################################