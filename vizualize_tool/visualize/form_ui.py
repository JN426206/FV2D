# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui',
# licensing of 'form.ui' applies.
#
# Created: Mon Apr 11 12:34:03 2022
#      by: pyside2-uic  running on PySide2 5.13.2
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_Widget(object):
    def setupUi(self, Widget):
        Widget.setObjectName("Widget")
        Widget.resize(800, 600)
        self.gridLayout_2 = QtWidgets.QGridLayout(Widget)
        self.gridLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_input_pred = QtWidgets.QGroupBox(Widget)
        self.groupBox_input_pred.setObjectName("groupBox_input_pred")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_input_pred)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_input_pred)
        self.lineEdit.setEnabled(False)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.PB_input_file = QtWidgets.QPushButton(self.groupBox_input_pred)
        self.PB_input_file.setObjectName("PB_input_file")
        self.horizontalLayout.addWidget(self.PB_input_file)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_input_pred)
        self.label_pitch = QtWidgets.QLabel(Widget)
        self.label_pitch.setText("")
        self.label_pitch.setObjectName("label_pitch")
        self.verticalLayout.addWidget(self.label_pitch)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.toolButton = QtWidgets.QToolButton(Widget)
        self.toolButton.setEnabled(False)
        self.toolButton.setMaximumSize(QtCore.QSize(16777215, 25))
        self.toolButton.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.toolButton.setArrowType(QtCore.Qt.RightArrow)
        self.toolButton.setObjectName("toolButton")
        self.horizontalLayout_2.addWidget(self.toolButton)
        self.HSB_frames = QtWidgets.QScrollBar(Widget)
        self.HSB_frames.setEnabled(False)
        self.HSB_frames.setMaximumSize(QtCore.QSize(16777215, 25))
        self.HSB_frames.setOrientation(QtCore.Qt.Horizontal)
        self.HSB_frames.setObjectName("HSB_frames")
        self.horizontalLayout_2.addWidget(self.HSB_frames)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout.setStretch(1, 1)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Widget)
        QtCore.QMetaObject.connectSlotsByName(Widget)

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QtWidgets.QApplication.translate("Widget", "Widget", None, -1))
        self.groupBox_input_pred.setTitle(QtWidgets.QApplication.translate("Widget", "Plik wynikowy", None, -1))
        self.PB_input_file.setText(QtWidgets.QApplication.translate("Widget", "Wybierz plik", None, -1))
        self.toolButton.setText(QtWidgets.QApplication.translate("Widget", "...", None, -1))

