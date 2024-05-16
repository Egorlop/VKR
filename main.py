import sys
from PySide6 import QtWidgets
import PostureDetecting.CreateDataSet as cds
from UI.PostureDetector import MainWindow
import warnings
warnings.filterwarnings("ignore")
import GoogleDrive.UploadAndDownloadDrive as uadd

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())