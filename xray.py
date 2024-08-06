from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from PyQt5 import uic
from PyQt5.uic import loadUiType
import mysql.connector as con
from datetime import date
import math
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus.tables import Table
from reportlab.lib.units import inch
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.pagesizes import letter
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras.preprocessing import image

#########################################################
model = tf.keras.models.load_model('chest_xray_classifier.h5')

def preprocess_image(img_path):
    """Preprocess the image for prediction."""
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale pixel values to [0, 1]
    return img_array

def predict_image(img_path):
    """Predict if the image is Normal or PNEUMONIA."""
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        return "PNEUMONIA"
    else:
        return "Normal"
###############################################
ui, _ = loadUiType('xray.ui')

class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget.tabBar().setVisible(False)
        self.menubar.setVisible(False)
        self.b01.clicked.connect(self.login)
        self.b11.clicked.connect(self.open_image)


    def login(self):
        un = self.tb01.text()
        pw = self.tb02.text()
        if(un=="" and pw==""):
            self.menubar.setVisible(True)
            self.tabWidget.setCurrentIndex(1)
        else:
            QMessageBox.information(self, "GFR Ccalculation System", "Invalid Login")
            self.tb03.setText("Invalid Admin Login")

##############################################################
    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg)', options=options)
        if file_name:
            self.display_image(file_name)
            prediction = predict_image(file_name)
            self.lb11.setText(f'Prediction: {prediction}')

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.gv01.setPixmap(pixmap.scaled(self.gv01.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))








##############################################################

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
    

            
