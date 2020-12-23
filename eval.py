# -*- coding: utf-8 -*-
# импортируйте все библиотеки, которые вам понадобятся
import cv2
import tensorflow as tf
import numpy as np

def standardize_input(image):
    """Приведение изображений к стандартному виду.
        Входные данные: изображение
        Выходные данные: стандартизированное изображений.
    """
    standard_im = cv2.imread(image)
    standard_im = cv2.cvtColor(standard_im, cv2.COLOR_BGR2GRAY)

    ## TODO: Если вы хотите преобразовать изображение в формат, одинаковый для всех изображений, сделайте это здесь.
    return standard_im

# Пропишите путь к вашей модели
MODEL_FILE_NAME = './'

def load_final_model():
    """ Функция осуществляет загрузку модели машинного обучения из файла MODEL_FILE_NAME.
        Выходные параметры: загруженная модель
    """
    ## TODO: Функция загрузки модели
    model1 = tf.keras.models.load_model("d1.h5")
    model2 = tf.keras.models.load_model("d2.h5")
    model3 = tf.keras.models.load_model("d3.h5")

    return model1, model2, model3

def predict(image, model):
    """ Функция осуществляет подачу данных на вход модели и возвращает вектор формата [1, 2]. Длина вектора может быть от 1 до 3.
         Входные параметры: изображение
         Выходные параметры: вектор с определёнными цифрами на изображении.
    """
    ## TODO: Функция предсказания вектора цифр по заданной картинке
    image = np.reshape(image, (image[0], image[1], 1))
    data = tf.convert_to_tensor([tf.cast(image, tf.float64) / 127.5 - 1.0])
    digit1 = model[0].predict(data)[0][0]
    digit2 = model[1].predict(data)[0][0]
    digit3 = model[2].predict(data)[0][0]
    ans = []
    if digit1 > 0.5:
        ans.append(1)
    if digit2 > 0.5:
        ans.append(2)
    if digit3 > 0.5:
        ans.append(3)

    return ans
