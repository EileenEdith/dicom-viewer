import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QScrollArea,
    QComboBox, QGraphicsBlurEffect, QSlider
)
from PyQt5.QtGui import QPixmap, QImage, QTransform
from PyQt5.QtCore import Qt

from ml.tf_classifier import TumorClassifierTF
from ml.gradcam_tf import generate_gradcam
from imaging.dicom_loader import load_dicom
from imaging.image_ops import adjust_brightness


LAST_CONV_LAYER = "Top_Conv_Layer"


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Viewer")

        self.classifier = TumorClassifierTF("models/brain_tumor_model.h5")

        self.current_path = None
        self.original_pixmap = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        image_layout = QHBoxLayout()
        self.image_label = QLabel()
        self.gradcam_label = QLabel()

        scroll1 = QScrollArea()
        scroll1.setWidget(self.image_label)
        scroll2 = QScrollArea()
        scroll2.setWidget(self.gradcam_label)

        image_layout.addWidget(scroll1)
        image_layout.addWidget(scroll2)

        # Buttons
        open_btn = QPushButton("Open Image")
        open_btn.clicked.connect(self.open_image)

        predict_btn = QPushButton("Predict Tumor + GradCAM")
        predict_btn.clicked.connect(self.predict_and_gradcam)

        flip_btn = QPushButton("Flip")
        flip_btn.clicked.connect(self.flip_image)

        rotate_l_btn = QPushButton("Rotate Left")
        rotate_l_btn.clicked.connect(lambda: self.rotate_image(90))

        rotate_r_btn = QPushButton("Rotate Right")
        rotate_r_btn.clicked.connect(lambda: self.rotate_image(-90))

        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(lambda: self.zoom(1.2))

        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(lambda: self.zoom(1 / 1.2))

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.valueChanged.connect(self.update_brightness)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["No Filter", "Blur"])
        self.filter_combo.currentIndexChanged.connect(self.apply_filter)

        self.result_label = QLabel("Prediction: -")

        # Layout assemble
        main_layout.addLayout(image_layout)
        main_layout.addWidget(open_btn)
        main_layout.addWidget(predict_btn)
        main_layout.addWidget(flip_btn)
        main_layout.addWidget(rotate_l_btn)
        main_layout.addWidget(rotate_r_btn)
        main_layout.addWidget(zoom_in_btn)
        main_layout.addWidget(zoom_out_btn)
        main_layout.addWidget(QLabel("Brightness"))
        main_layout.addWidget(self.brightness_slider)
        main_layout.addWidget(self.filter_combo)
        main_layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # ---------- Image loading ----------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.dcm *.png *.jpg)"
        )
        if not path:
            return

        self.current_path = path
        self.brightness_slider.setValue(0)

        if path.endswith(".dcm"):
            img, _ = load_dicom(path)
            img = adjust_brightness(img, 0)
            qimg = QImage(
                img.data, img.shape[1], img.shape[0],
                img.strides[0], QImage.Format_Grayscale8
            )
        else:
            img = Image.open(path).convert("RGB")
            img = np.array(img)
            qimg = QImage(
                img.data, img.shape[1], img.shape[0],
                img.strides[0], QImage.Format_RGB888
            )

        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)
        self.image_label.adjustSize()
        self.original_pixmap = pixmap
        self.gradcam_label.clear()

    # ---------- Image ops ----------
    def update_brightness(self, value):
        if not self.current_path:
            return

        if self.current_path.endswith(".dcm"):
            img, _ = load_dicom(self.current_path)
            img = adjust_brightness(img, value)
            qimg = QImage(
                img.data, img.shape[1], img.shape[0],
                img.strides[0], QImage.Format_Grayscale8
            )
            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap)
            self.original_pixmap = pixmap

    def flip_image(self):
        if self.image_label.pixmap():
            self.image_label.setPixmap(
                self.image_label.pixmap().transformed(QTransform().scale(-1, 1))
            )

    def rotate_image(self, angle):
        if self.image_label.pixmap():
            self.image_label.setPixmap(
                self.image_label.pixmap().transformed(QTransform().rotate(angle))
            )

    def zoom(self, factor):
        pixmap = self.image_label.pixmap()
        if pixmap:
            self.image_label.setPixmap(
                pixmap.scaled(pixmap.size() * factor, Qt.KeepAspectRatio)
            )

    def apply_filter(self):
        if not self.original_pixmap:
            return

        if self.filter_combo.currentText() == "Blur":
            blur = QGraphicsBlurEffect()
            blur.setBlurRadius(6)
            self.image_label.setGraphicsEffect(blur)
        else:
            self.image_label.setGraphicsEffect(None)
            self.image_label.setPixmap(self.original_pixmap)

    # ---------- ML ----------
    def predict_and_gradcam(self):
        if not self.current_path:
            return

        label, probs = self.classifier.predict(self.current_path)
        self.result_label.setText(f"Prediction: {label}")

        img = tf.keras.preprocessing.image.load_img(
            self.current_path, target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        class_idx = np.argmax(probs)
        heatmap = generate_gradcam(
            self.classifier.model,
            img_array,
            LAST_CONV_LAYER,
            class_idx
        )

        self.show_gradcam(heatmap)

    def show_gradcam(self, heatmap):
        img = Image.open(self.current_path).convert("RGB")
        img = img.resize((heatmap.shape[1], heatmap.shape[0]))
        img = np.array(img)

        heatmap = np.uint8(255 * heatmap)
        jet = plt.cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        cam = jet_colors[heatmap]

        cam = (cam * 0.4 + img).astype(np.uint8)
        qimg = QImage(
            cam.data, cam.shape[1], cam.shape[0],
            cam.strides[0], QImage.Format_RGB888
        )
        self.gradcam_label.setPixmap(QPixmap.fromImage(qimg))
        self.gradcam_label.adjustSize()