from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def authenticate_gdrive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

def download_file_from_drive(file_id, save_path="Data/Documents"):
    drive = authenticate_gdrive()
    file = drive.CreateFile({'id': file_id})
    file.FetchMetadata()
    file_name = file['title']
    file.GetContentFile(os.path.join(save_path, file_name))
    return os.path.join(save_path, file_name)
