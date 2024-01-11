from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QWidget, QVBoxLayout
from PySide6.QtGui import QPixmap, QAction
from PySide6 import QtCore

from noaa_decoder import NOAADecoder
from main_page import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.OpenFileAction = self.findChild(QAction, 'OpenFileAction')
        self.FileProcessingButton = self.findChild(QPushButton, 'FileProcessingButton')
        self.FileNameTitle = self.findChild(QLabel, 'FileNameTitle')
        self.SatelliteImage = self.findChild(QLabel, 'SatelliteImage')
        self.Graphic = self.findChild(QWidget, 'Graphic')

        self.OpenFileAction.triggered.connect(self.clicker)

        self.show()

    
    def clicker(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'WAV Files (*.wav)')
        
        if self.fname:
            satellite_data = NOAADecoder(self.fname, 60)

            layout = QVBoxLayout(self.Graphic)
            layout.addWidget(satellite_data.get_signal_plot())

            self.FileNameTitle.setText(satellite_data.img_save_path)

            self.pixmap = QPixmap(satellite_data.img_save_path)
            self.pixmap = self.pixmap.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
            self.SatelliteImage.setPixmap(self.pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec())
