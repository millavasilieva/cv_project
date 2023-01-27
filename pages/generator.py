import streamlit as st
import io

from PIL import Image

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import ToTensor,Compose,Resize,Normalize,CenterCrop

latent_size = 64
class GeneratorGAN(nn.Module):
    def init(self):
        super(GeneratorGAN, self).init()
        self.fc1 = nn.Linear(latent_size + 10, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 784)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # print(x.shape)
        # print(x.size(0))  
        # print(x.view(x.size(0), -1))
        # print(x.view(x.size(0), -1).shape)     
        return x

def load_model():
    generator = GeneratorGAN()
    generator.load_state_dict(torch.load('CGAN_generator.pt'))
    
    return generator

# def predict(model, image):
    preprocess = Compose([
        Resize((64,64)),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5]),
    ])
    # input_tensor = preprocess(image)
    # # input_batch = input_tensor.unsqueeze(0)
    
    # st.write("Я думаю, что это...")
    # class_ = model(input_tensor.unsqueeze(0)).argmax().item()
     

def main():
    st.title('Генерация изображений на базе датасета MNIST')
    st.write('Выберите цифру')
    # option = st.radio("Цифра.", ['0','1', '2', '3', '4', '5', '6', '7', '8', '9'])
    option = st.number_input('Введите цифру от 0 до 9', min_value=0, max_value=9, value=1)
    if option not in range(0,10):
        print('Введите число от 0 до 9')
    else:
        y = torch.tensor(option)
        y = torch.nn.functional.one_hot(y, num_classes = 10)
        y = y.unsqueeze(0)
    model = load_model()
    # categories = load_labels()
    image = load_image()
    result = st.button('Нажми для предсказания')
    if result:
        st.write('Обрабатываем результаты...')
        predict(model, image)


if __name__ == '__main__':
    main()