import os
import cv2
import torch
import numpy as np
import mysql.connector

from PySide6.QtGui import QIcon
from PySide6 import QtWidgets, QtCore, QtGui
from ultralytics import YOLO


class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 連線 MySQL 資料庫
        self.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="yolo_abnormal_fish_detection"
        )
        self.cursor = self.db.cursor()

        self.init_gui()
        self.model = None
        self.timer = QtCore.QTimer()
        self.timer1 = QtCore.QTimer()
        self.cap = None
        self.video = None
        self.timer.timeout.connect(self.camera_show)
        self.timer1.timeout.connect(self.video_show)

    def init_gui(self):
        self.setFixedSize(960, 440)
        self.setWindowTitle('Fish Behavior Monitoring System')
        self.setWindowIcon(QIcon(""))

        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        topLayout = QtWidgets.QHBoxLayout()
        self.oriVideoLabel = QtWidgets.QLabel(self)
        self.detectlabel = QtWidgets.QLabel(self)
        self.oriVideoLabel.setMinimumSize(448, 336)
        self.detectlabel.setMinimumSize(448, 336)
        self.oriVideoLabel.setStyleSheet('border:1px solid #D7E2F9;')
        self.detectlabel.setStyleSheet('border:1px solid #D7E2F9;')

        topLayout.addWidget(self.oriVideoLabel)
        topLayout.addWidget(self.detectlabel)

        mainLayout.addLayout(topLayout)

        # 下方按鈕
        groupBox = QtWidgets.QGroupBox(self)
        bottomLayout = QtWidgets.QVBoxLayout(groupBox)
        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QHBoxLayout()
        self.selectModel = QtWidgets.QPushButton('Select Model')
        self.selectModel.setFixedSize(100, 50)
        self.selectModel.clicked.connect(self.load_model)

        self.openVideoBtn = QtWidgets.QPushButton('Open Video')
        self.openVideoBtn.setFixedSize(100, 50)
        self.openVideoBtn.clicked.connect(self.start_video)
        self.openVideoBtn.setEnabled(False)

        self.openCamBtn = QtWidgets.QPushButton('Open Camera')
        self.openCamBtn.setFixedSize(100, 50)
        self.openCamBtn.clicked.connect(self.start_camera)

        self.stopDetectBtn = QtWidgets.QPushButton('Stop Detect')
        self.stopDetectBtn.setFixedSize(100, 50)
        self.stopDetectBtn.setEnabled(False)
        self.stopDetectBtn.clicked.connect(self.stop_detect)

        btnLayout.addWidget(self.selectModel)
        btnLayout.addWidget(self.openVideoBtn)
        btnLayout.addWidget(self.openCamBtn)
        btnLayout.addWidget(self.stopDetectBtn)

        bottomLayout.addLayout(btnLayout)

    def start_camera(self):
        self.timer1.stop()
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            self.timer.start(50)
        self.stopDetectBtn.setEnabled(True)

    def camera_show(self):
        ret, frame = self.cap.read()
        if ret:
            if self.model is not None:
                frame = cv2.resize(frame, (448, 352))
                results = self.model(frame, imgsz=[448, 352], device='cuda' if torch.cuda.is_available() else 'cpu')

                frame1 = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
                frame1 = QtGui.QImage(frame1.data, frame1.shape[1], frame1.shape[0], QtGui.QImage.Format_RGB888)
                self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(frame1))

                self.save_results_to_db(results)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(frame))
            self.oriVideoLabel.setScaledContents(True)

    def start_video(self):
        if self.timer.isActive():
            self.timer.stop()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video", filter="*.mp4 *.avi *.mov")
        if fileName:
            self.video = cv2.VideoCapture(fileName)
            if self.video.isOpened():
                fps = self.video.get(cv2.CAP_PROP_FPS)
                self.timer1.start(int(1000 / fps))
        self.stopDetectBtn.setEnabled(True)

    def video_show(self):
        ret, frame = self.video.read()
        if ret:
            if self.model is not None:
                frame = cv2.resize(frame, (448, 352))
                results = self.model(frame, imgsz=[448, 352], device='cuda' if torch.cuda.is_available() else 'cpu')

                frame1 = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
                frame1 = QtGui.QImage(frame1.data, frame1.shape[1], frame1.shape[0], QtGui.QImage.Format_RGB888)
                self.detectlabel.setPixmap(QtGui.QPixmap.fromImage(frame1))

                self.save_results_to_db(results)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.oriVideoLabel.setPixmap(QtGui.QPixmap.fromImage(frame))
            self.oriVideoLabel.setScaledContents(True)
        else:
            self.video.release()
            self.timer1.stop()

    def load_model(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model", filter="*.pt")
        if fileName.endswith('.pt'):
            self.model = YOLO(fileName)
        else:
            print("Please select a valid model")

        self.openVideoBtn.setEnabled(True)
        self.stopDetectBtn.setEnabled(True)

    def stop_detect(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.timer1.isActive():
            self.timer1.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.video is not None:
            self.video.release()
            self.video = None

    def save_results_to_db(self, results):
        for result in results:
            for box in result.boxes:
                class_name = "abnormal"  # 類別名稱根據模型進行修改
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                sql = "INSERT INTO detections (class_name, confidence, x1, y1, x2, y2) VALUES (%s, %s, %s, %s, %s, %s)"
                values = (class_name, confidence, x1, y1, x2, y2)
                self.cursor.execute(sql, values)
                self.db.commit()

    def closeEvent(self, event):
        self.db.close()
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication()
    window = MyWindow()
    window.show()
    app.exec()
