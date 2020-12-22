# -*- coding: utf-8 -*-
import os

import numpy as np
from eval import standardize_input, predict, load_final_model
import csv
import ast


def csv_reader(file_obj):
    """
    Read a csv file
    """
    reader = csv.reader(file_obj)
    image_paths = []
    image_labels = []
    for row in reader:
        img = ",".join(row)
        path = img.split(";")[0]
        image_paths.append(path)
        label = img.split(";")[1]
        image_labels.append(ast.literal_eval(label))

    return image_paths, image_labels



def load_data():
    IMAGE_DIR_VALIDATION = "pictures_numeric_val/"


    data = []
    csv_path = "validate.csv"

    with open(csv_path, "r") as f_obj:
        imagePaths, labels = csv_reader(f_obj)

    for imagePath in imagePaths:
        print(imagePath)
        data.append(standardize_input(IMAGE_DIR_VALIDATION + imagePath))

    return data, labels


def get_predictions(data, model):
    if model is not None:
        predictions = [predict(image, model) for image in data]
    else:
        predictions = [[2, 1] for x in range(len(data))]
        print('constant_predictions', predictions)

    return predictions

def get_score(labels, predictions):
    score = 0
    max_points = len([label for vec in labels for label in vec])
    for label, pred in zip(labels, predictions):
        if label == pred:
           score += len(pred)

    return float(score / max_points)

def main():
    data, labels = load_data()
    try:
        model = load_final_model()
        print('model', model)
        predictions = get_predictions(data, model)
        print(predictions)
    except Exception as e:
        print(e)
        print('Модель не подгружена, используем константный классификатор')
        model = None
        predictions = get_predictions(data, model)

    score = get_score(labels, predictions)

    try:
        score = get_score(labels, predictions)
        print('Точность модели:', score)
    except Exception as e:
        print('Ошибка:', e)

    file = open("score.txt", "w")
    file.write(str(score))
    file.close()


if __name__ == '__main__':
    main()
