import streamlit as st
from streamlit_paste_button.__init__ import paste_image_button as pbutton
from io import StringIO
import pandas as pd
import cv2 as cv
import imutils
import os
import numpy as np
from skimage.util.shape import view_as_blocks
from skimage import io, transform
import keras
from tensorflow.keras import backend as K
SQUARE_SIZE = 40
    
def weighted_categorical_crossentropy(weights):

    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss



def model():
    names = ["1.hdf5", "2.hdf5", "3.hdf5", "4.hdf5", "5.hdf5"]
    names = list(map(lambda st: "./weights/"+st, names))
    models=[]
    for model_name in names:
        models.append(keras.models.load_model(model_name, custom_objects={'loss':weighted_categorical_crossentropy(np.array([1/(0.30*4), 1/(0.20*4), 1/(0.20*4), 1/(0.20*4), 1/1,  1/(0.10*4), 1/(0.30*4), 1/(0.20*4), 1/(0.20*4), 1/(0.20*4), 1/1,  1/(0.10*4), 1/(64-10)]))}))
    input_layer=keras.layers.Input(shape=(40, 40, 3))
    xs=[model(input_layer) for model in models]
    out=keras.layers.Add()(xs)
    
    model=keras.models.Model(inputs=[input_layer], outputs=out)
    return model



def process_image(img):
    print(type(img))
    downsample_size = SQUARE_SIZE*8
    square_size = SQUARE_SIZE   
    # img_read = io.imread(img)
    img_read = img
    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)



def fen_from_onehot(one_hot):
    piece_symbols = 'prbnkqPRBNKQ'
    output = ''
    for j in range(8):
        for i in range(8):
            if(one_hot[j][i] == 12):
                output += ' '
            else:
                output += piece_symbols[one_hot[j][i]]
        if(j != 7):
            output += '-'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output



def predict(image):
    model_instance = model()
    print(image.shape)
    pred = model_instance.predict(image).argmax(axis=1).reshape(-1, 8, 8)
    fen = fen_from_onehot(pred[0])
    print(fen)
    return fen




def extract(image):
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    img = cv.Canny(image, 10, 170)
    # Gaussian Blurring
    img = cv.GaussianBlur(img, (5, 5), 0)
    # croppedlis = []
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > (img.shape[0] * img.shape[1] * 1 / 12):  # IMPORTANT:add square condition, multiplicant was 1/12
            # cv.drawContours(image, cnt, -1, (255, 100, 100), 3)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, False)  # getting the boounding box
            x, y, w, h = cv.boundingRect(approx)
            # croppedlis.append(image[y:y + h, x:x + w])
            return image[y:y + h, x:x + w]


def open_new_window(url):
    new_tab = f"<a target='_blank' href='{url}' style='font-size: 40px;'>Open Board Editor</a>"
    st.markdown(new_tab, unsafe_allow_html=True)


def fen_web_page(fen):
    predicate = "https://lichess.org/editor/"
    fen = fen.replace("-", "/")
    return predicate+fen

def app():
    st.title("Chess Position Scanner")
    paste_result = pbutton("📋 Paste an image")
    if paste_result.image_data is not None:
        st.write('Pasted image:')
        image = extract(paste_result.image_data)
        st.image(image)
        with st.spinner("Generating FEN Link..."):
            image = process_image(image)
                
            fen = predict(image)
        
        open_new_window(fen_web_page(fen))
# https://lichess.org/analysis/standard/6q1/pp2rp2/2p4p/5Q1B/1P1q4/P1b5/5PPP/3R2K1  
# https://lichess.org/editor/6q1/pp2rp2/2p4p/5Q1B/1P1q4/P1b5/5PPP/3R2K1

#2fRAUa6scCeOsywTESORW0226f1_2CQ8zoPguRqmgCCGjYKx5
if __name__ == "__main__":
    app()