# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys
import cv2
import numpy as np

from PySide2.QtWidgets import QApplication, QWidget, QFileDialog
from PySide2.QtCore import QFile, QDir, qDebug, QTimer
from PySide2 import QtCore
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QImage, QPixmap
from form_ui import Ui_Widget

from detected_object import DetectedObject

class Widget(QWidget):
    resized = QtCore.Signal()
    def __init__(self):
        super(Widget, self).__init__()
        self.ui_form = Ui_Widget()
        self.ui_form.setupUi(self)
        self.pitch = None
        self.load_pitch()
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self.nextFrameTimer)
        self.playOn = False
        self.ui_form.PB_input_file.clicked.connect(self.onPBinputFile)
        self.resized.connect(self.draw_pitch)
        self.ui_form.HSB_frames.valueChanged.connect(self.onHSB_framesValueChanged)
        self.ui_form.toolButton.clicked.connect(self.onPlayButton)
        self.frames = {}
        self.current_frame_id = 1

    def resizeEvent(self, event):
        self.resized.emit()
        return super(Widget, self).resizeEvent(event)

    def load_ui(self):
        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        main_ui = loader.load(ui_file, self)
        ui_file.close()
        return main_ui

    def onPBinputFile(self):
        self.frames = {}
        self.ui_form.HSB_frames.setEnabled(False)
        self.ui_form.toolButton.setEnabled(False)
        self.ui_form.lineEdit.clear()
        self.draw_pitch()
        # frame_id  object_class    score   bb_cx   bb_cy   bb_height   bb_width    trackId colorRED    colorGREEN  colorBlue   pitchx  pitchy
        input_file_path = QFileDialog.getOpenFileName(self, self.tr("Wybierz plik wejściowy"), QDir.homePath(), self.tr("(*.csv *.tsv)"))
        if len(input_file_path[0]) == 0:
            return
        self.ui_form.lineEdit.setText(input_file_path[0])
        seperator = "\t"
        if input_file_path[0].split(".")[-1].lower() == "csv":
            seperator = ";"
        with open(input_file_path[0], 'r') as input_file:
            last_frame_id = 1
            detected_objects = []
            frame_id = 1
            for line in input_file.readlines():
                data = line.replace("\n", "").split(seperator)
                frame_id = int(data[0])
                if frame_id != last_frame_id:
                    self.frames[last_frame_id] = detected_objects
                    detected_objects = []
                last_frame_id = frame_id
                object_class = int(data[1])
                score = float(data[2])
                bbox = [int(data[3]), int(data[4]), int(data[3])-int(data[5]), int(data[4])-int(data[6])]
                trackId = None if len(data[7]) == 0 else int(data[7])
                color = (255, 0, 0) if len(data[8]) == 0 else (int(data[8]), int(data[9]), int(data[10]))
                pitchxy = None if len(data[11]) == 0 else (float(data[11]), float(data[12]))
                if not pitchxy:
                    continue
                detected_objects.append(DetectedObject(bbox, object_class, score, color, trackId=trackId, pitchxy=pitchxy))                
            self.frames[last_frame_id] = detected_objects
        if len(self.frames) > 0:
            self.ui_form.HSB_frames.setEnabled(True)
            self.ui_form.HSB_frames.setMinimum(1)
            self.ui_form.HSB_frames.setMaximum(len(self.frames))
            self.ui_form.toolButton.setEnabled(True)
            self.draw_pitch()


    def draw_on_pitch(self, detected_objects, pitch):
        pitch_height = pitch.shape[0]
        pitch_width = pitch.shape[1]
        for index, objectDetected in enumerate(detected_objects):
            bbox = objectDetected.bbox
            if not objectDetected.color:
                color = (255, 0, 0)
            else:
                color = (int(objectDetected.color[0]), int(objectDetected.color[1]), int(objectDetected.color[2]))
                cv2.circle(pitch, (int(objectDetected.pitchxy[0]*pitch_width), int(objectDetected.pitchxy[1]*pitch_height)), 5, color, -1)
                cv2.circle(pitch, (int(objectDetected.pitchxy[0]*pitch_width), int(objectDetected.pitchxy[1]*pitch_height)), 6, (0, 0, 0), 1)
        return pitch

    def draw_pitch(self):
        pitch = Widget.image_resize(self.pitch, self.ui_form.label_pitch.width())
        if len(self.frames) > 0:
            pitch = self.draw_on_pitch(self.frames[self.current_frame_id], pitch)
        height, width, channel = pitch.shape
        bytesPerLine = 3 * width
        qpitch = QImage(pitch.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.ui_form.label_pitch.setPixmap(QPixmap(qpitch))


    @staticmethod
    def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation = inter)
        return resized


    def load_pitch(self):
        self.pitch = cv2.imread("resources/world_cup_template.png")
        self.pitch = cv2.resize(self.pitch, (320, 180))

    def onPlayButton(self):
        if self.current_frame_id == len(self.frames):
            self.current_frame_id = 1
            self.ui_form.HSB_frames.setValue(self.current_frame_id)
        if not self.playOn:
            self.playOn = True
            self.ui_form.toolButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            self.ui_form.toolButton.setText("||")
            self.timer.start()
        else:
            self.playOn = False
            self.ui_form.toolButton.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
            self.timer.stop()

    def nextFrameTimer(self):
        if self.playOn and self.current_frame_id+1 <= len(self.frames):
            self.current_frame_id += 1
            self.ui_form.HSB_frames.setValue(self.current_frame_id)
            self.draw_pitch()
        elif self.playOn:
            self.timer.stop()
            self.playOn = False
            self.ui_form.toolButton.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)

    def onHSB_framesValueChanged(self, pos):
        self.current_frame_id = pos
        self.draw_pitch()


if __name__ == "__main__":
    app = QApplication([])
    widget = Widget()
    widget.show()
    sys.exit(app.exec_())
