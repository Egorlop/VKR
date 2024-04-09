import sys
from PySide6 import QtWidgets
import PostureDetecting.CreateDataSet as cds
from UI.PostureDetector import PostureDetector


def main():
    #NewUserDataset()
    #uadd.DownloadFromDrive()
    datasetxl, datasetpd, datasetfeat = cds.ReadFromExcel('All')
    testdata,testfeat, traindata, validatedata, trainfeat, validatefeat = SplitDataset(datasetxl, datasetfeat)
    classificator=PickClassificator(testdata,testfeat,traindata,trainfeat,validatedata,validatefeat)
    cds.LiveTest(classificator)
    #ws.WriteScatter(datasetpd)
    #km.KnnMeshgrid()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = PostureDetector()
    win.show()
    sys.exit(app.exec())