# -*- coding: utf-8 -*-
"""Implementação do Aumento no conjunto de Imagens
"""

import os, sys
import cv2
from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob, os
import PIL
from PIL import Image
import imutils
from glob import glob
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

def redimencionar(path, maxsize):
        img = Image.open(path)
        img = img.resize((277, 277))
        return np.asarray(img)

i=0
os.chdir('/content/drive/MyDrive/UltimoPrometo/pizedouros')
for file in glob(os.path.join(os.getcwd(),'*.jpg')):
  maxsize =277, 277
  img = Image.open(file)
  img = redimencionar(file, maxsize)

  #converetendo para RGB
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
  
  imagem90 = imutils.rotate(img, angle=90)
  imagem180 =  imutils.rotate(img, angle=180)
  imagem270 =  imutils.rotate(img, angle=270)
  flippe90= cv2.flip(imagem90, 0) 
  flippe180= cv2.flip(imagem180, 0)
  flippe270= cv2.flip(imagem270, 0)
  #plt.imshow(img, cmap=plt.cm.gray)
  #plt.show()
  #plt.imshow(imagem90, cmap=plt.cm.gray)
  #plt.show()
  #plt.imshow(flippedimage, cmap=plt.cm.gray)
  #plt.show 

  cv2.imwrite('/content/drive/MyDrive/UltimoPrometo/todas/Pizedouros__%04i.jpg'%i,  img)
  cv2.imwrite('/content/drive/MyDrive/UltimoPrometo/todas/Pizedouros__90_%04i.jpg' %i,  imagem90)
  cv2.imwrite('/content/drive/MyDrive/UltimoPrometo/todas/Pizedouros__180_%04i.jpg' %i,  imagem180)
  cv2.imwrite('/content/drive/MyDrive/UltimoPrometo/todas/Pizedouros__270_%04i.jpg' %i,  imagem270)
  cv2.imwrite('/content/drive/MyDrive/UltimoPrometo/todas/Pizedouros__90F_%04i.jpg' %i,  flippe90)
  cv2.imwrite('/content/drive/MyDrive/UltimoPrometo/todas/Pizedouros__180F_%04i.jpg' %i, flippe180)
  cv2.imwrite('/content/drive/MyDrive/UltimoPrometo/todas/Pizedouros__270F_%04i.jpg' %i, flippe270)
  i+=1

import imutils
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
i=0
#for fn in img_names:
path = '/content/drive/MyDrive/imagem'
files = glob(path+'*jpg')
for file in files:
  maxsize =277, 277
  img = Image.open(file)
  #width, height = img.size
  (new_width, new_heigth) =( 277, 277)
  img = img.resize((round(new_width),round(new_heigth)), Image.ANTIALIAS) 
  img= np.asarray(img)
  #img = img.resize((277, 277))
  #img = jpeg_to_8_bit_greyscale(file, maxsize)
  imagem90 = imutils.rotate(img, angle=90)
  imagem180 =  imutils.rotate(img, angle=180)
  imagem270 =  imutils.rotate(img, angle=270)
  #im_flip = ImageOps.flip(imagem90) #inverte horizontalmente
  flippe90= cv2.flip(imagem90, 0) 
  flippe180= cv2.flip(imagem180, 0)
  flippe270= cv2.flip(imagem270, 0)
  #plt.imshow(img, cmap=plt.cm.gray)
  #plt.show()
  #plt.imshow(imagem90, cmap=plt.cm.gray)
  #plt.show()
  #plt.imshow(flippedimage, cmap=plt.cm.gray)
  #plt.show 
  cv2.imwrite('/content/drive/MyDrive/imagem/imagem.jpg',image)
  #cv2.imwrite('/content/drive/MyDrive/Rotacionadas/Coccinellidae/Coccinellidae_%04i.jpg'%i,  img)
  #cv2.imwrite('/content/drive/MyDrive/Rotacionadas/Coccinellidae/Coccinellidae_90_%04i.jpg' %i,  imagem90)
  #cv2.imwrite('/content/drive/MyDrive/Rotacionadas/Coccinellidae/Coccinellidae_180_%04i.jpg' %i,  imagem180)
  #cv2.imwrite('/content/drive/MyDrive/Rotacionadas/Coccinellidae/Coccinellidae_270_%04i.jpg' %i,  imagem270)
  #cv2.imwrite('/content/drive/MyDrive/Rotacionadas/Coccinellidae/Coccinellidae_90F_%04i.jpg' %i,  flippe90)
  #cv2.imwrite('/content/drive/MyDrive/Rotacionadas/Coccinellidae/Coccinellidae_180F_%04i.jpg' %i, flippe180)
  #cv2.imwrite('/content/drive/MyDrive/Rotacionadas/Coccinellidae/Coccinellidae_270F_%04i.jpg' %i, flippe270)
  i+=1

