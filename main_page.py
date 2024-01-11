# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_page.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QMenu,
    QMenuBar, QPushButton, QSizePolicy, QStatusBar,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1111, 881)
        self.OpenFileAction = QAction(MainWindow)
        self.OpenFileAction.setObjectName(u"OpenFileAction")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.SatelliteImage = QLabel(self.centralwidget)
        self.SatelliteImage.setObjectName(u"SatelliteImage")
        self.SatelliteImage.setGeometry(QRect(480, 10, 591, 511))
        self.SatelliteImage.setAlignment(Qt.AlignCenter)
        self.Graphic = QWidget(self.centralwidget)
        self.Graphic.setObjectName(u"Graphic")
        self.Graphic.setGeometry(QRect(480, 570, 621, 261))
        self.FileProcessingButton = QPushButton(self.centralwidget)
        self.FileProcessingButton.setObjectName(u"FileProcessingButton")
        self.FileProcessingButton.setGeometry(QRect(90, 160, 181, 61))
        self.FileNameTtile = QLabel(self.centralwidget)
        self.FileNameTtile.setObjectName(u"FileNameTitle")
        self.FileNameTtile.setGeometry(QRect(40, 60, 271, 61))
        self.FileNameTtile.setLayoutDirection(Qt.LeftToRight)
        self.FileNameTtile.setAlignment(Qt.AlignCenter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1111, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.OpenFileAction)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.OpenFileAction.setText(QCoreApplication.translate("MainWindow", u"Open File", None))
        self.SatelliteImage.setText(QCoreApplication.translate("MainWindow", u"Satellite image", None))
        self.FileProcessingButton.setText(QCoreApplication.translate("MainWindow", u"File Processing", None))
        self.FileNameTtile.setText(QCoreApplication.translate("MainWindow", u"File Name", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
    # retranslateUi

