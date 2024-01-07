from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from dialog import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.pushButton = self.findChild(QPushButton, 'pushButton')
        self.label = self.findChild(QLabel, 'label')


        self.pushButton.clicked.connect(self.clicker)

        self.show()

    
    def clicker(self):
        #self.label.setText('You Click!!!')

        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*)')

        if fname:
            self.label.setText(fname)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    sys.exit(app.exec())
