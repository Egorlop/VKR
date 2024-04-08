import time

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import datetime
import os
def UploadToDrive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    current_datetime = datetime.datetime.now()
    filename = str(current_datetime.date())+'_'+str(current_datetime.time())[:8].replace(':','-')+'_'+'DataSet.xlsx'
    file = drive.CreateFile({'title': filename,
                              'parents': [{'id': '1nxi08WrghEy712GQeHmqrO3DbXM-HEvw'}]})
    file.SetContentFile('D:\\pythonProject\\datasets\\CustomDataset.xlsx')
    file.Upload()
    print(f'File {filename} loaded and deleted from local dir')

def DownloadFromDrive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    file_list = drive.ListFile({'q': "'1nxi08WrghEy712GQeHmqrO3DbXM-HEvw' in parents and trashed=false"}).GetList()
    for file in file_list:
        print('Title: %s, ID: %s' % (file['title'], file['id']))
        file2 = drive.CreateFile({'id': file['id']})
        file2.GetContentFile('D:\\pythonProject\\datasets\\'+f"{file['title']}")