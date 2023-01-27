import streamlit as st
import io

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.nn import functional as F
from torchvision.transforms import ToTensor,Compose,Resize,Normalize,CenterCrop

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.SELU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=2),
            nn.BatchNorm2d(8),
            nn.SELU()
            )
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True) #<<<<<< Bottleneck
        
        #decoder
        
        self.unpool = nn.MaxUnpool2d(2, 2)
        
        self.conv1_t = nn.Sequential(
            nn.ConvTranspose2d(8, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.SELU()
            )
        self.conv2_t = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
            )        

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, indicies = self.pool(x) # ⟸ bottleneck
        return x, indicies

    def decode(self, x, indicies):
        x = self.unpool(x, indicies)
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        return x

    def forward(self, x):
        latent, indicies = self.encode(x)
        out = self.decode(latent, indicies)      

        return out


def load_model():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('autoencoder.pt'))
    
    return model

def load_image():
    uploaded_file = st.file_uploader(label='Выберите черно-белый скан документа',type=['png'], accept_multiple_files=False)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def denoise(deviceoption,model, image):

    trans = T.Compose([
        T.ToTensor()])

    resize = T.Compose(
         [T.Resize((250, 500))])

    img_old = trans(image.convert("L")) 
    original_size = img_old.squeeze(0).shape
    img_pr = resize(img_old)
    to_orig_size = T.Compose([T.Resize(original_size)])
    results = to_orig_size(model(torch.unsqueeze(img_pr,0)))
    
    st.write("Очищаем документ..")

    col1,col2 = st.columns(2)
    with col1:
        img = img_old.squeeze(0).detach().cpu().numpy()
        st.image(img, caption= 'Оригинальное изображение',use_column_width='always')
    with col2:
        img_ = results.detach().cpu().numpy()[0][0]
        st.image(img_, caption= 'Очищенное изображение',use_column_width='always')


def main():
    st.title('Очистка документов от шумов')

    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Выберите вычислительное устройство.", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("Выберите вычислительное устройство.", ['cpu', 'cuda'], disabled = True, index=0)


    model = load_model()
    scr = st.sidebar.radio("Тип загрузки файла", ['Загрузка собственных файлов','Загрузка демо файлов'])


    if scr == 'Загрузка демо файлов':
        options = ['sample 1', 'sample 2', 'sample 3']
        for i in range (len(options)):
            image = Image.open(f'denoise_sample/sample{i+1}.png')
            result = st.button(f'Нажми демо очистки документа {i+1}')
            if result:
                # st.write('Обрабатываем результаты...')
                denoise(deviceoption,model, image)


    if scr == 'Загрузка собственных файлов':
        image = load_image()
        res = st.button('Нажми для очистки документа')
        if res:
            # st.write('Обрабатываем результаты...')
            denoise(deviceoption, model, image)


if __name__ == '__main__':
    main()