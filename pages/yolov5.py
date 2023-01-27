import streamlit as st
import torch
# from yolov5 import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
# import matplotlib.pyplot as plt
# import os
# import wget
# import time

## CFG
# cfg_model_path = "yolov5/models/roadobjects.pt" 

cfg_enable_url_download = True
if cfg_enable_url_download:
    url = "https://archive.org/download/roadobjects/roadobjects.pt" #Configure this if you set cfg_enable_url_download to True
    cfg_model_path = f"models/{url.split('/')[-1:][0]}" #config model path from url name
## END OF CFG


def imageInput(deviceoption,src):
    
    if src == 'Загрузите свои файлы':
        image_file = st.file_uploader("Загрузите картинку", type=['png', 'jpeg', 'jpg'])
        # col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            # with col1:
            st.image(img, caption= 'Загруженный файл',use_column_width='always')
               
            #call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='res_models/roadobjects.pt') 
            

            #--Display predicton
            # with col2:
            results = model(img)
            img = results.render()[0]
            st.image(img, caption='Предсказание модели', use_column_width='always')

    if src == 'Посмотреть демо':

        options = ['sample 1', 'sample 2', 'sample 3']

        for i in range (len(options)):
            image = Image.open(f'yolo_samples/sample{i+1}.jpg')
            result = st.button(f'Нажми демо детекции объектов на проезжей части {i+1}')

            if result:
                img = st.image(image, caption= 'Загруженный файл',use_column_width='always')
                model = torch.hub.load('ultralytics/yolov5', 'custom', path='res_models/roadobjects.pt') 
                results = model(image)
                img_ = results.render()[0]
                st.image(img_, caption='Предсказание модели', use_column_width='always')


def main():
    # -- Sidebar
    st.sidebar.title('Выберите тип загрузки')
    datasrc = st.sidebar.radio("Тип загрузки", ['Загрузите свои файлы','Посмотреть демо'])
    

    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Выберите вычислительное устройство.", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("Выберите вычислительное устройство.", ['cpu', 'cuda'], disabled = True, index=0)

    st.header('Roadobjects Detection Model Demo')
    st.subheader('Выбрать картинку')   
    imageInput(deviceoption, datasrc)

if __name__ == '__main__':
  
    main()
# @st.cache
# def loadModel():
#     start_dl = time.time()
#     model_file = wget.download('https://archive.org/download/roadobjects/roadobjects.pt', out="res_models/")
#     finished_dl = time.time()
#     print(f"Модель загружена, ETA:{finished_dl-start_dl}")
# loadModel()