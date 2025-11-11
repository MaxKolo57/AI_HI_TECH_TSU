# main.py
import sys
from PyQt5.QtWidgets import QApplication
from ui_mainwindow import OrthoSegApp


def main():
    app = QApplication(sys.argv)
    w = OrthoSegApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
