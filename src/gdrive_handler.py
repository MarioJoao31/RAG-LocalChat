from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import re

def authenticate_gdrive():
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile("client_secret.json")
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

def extract_folder_id(folder_url_or_id):
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', folder_url_or_id)
    if match:
        print("debugg matchID:", match.group(1))
        return match.group(1)
    else:
        print("debugg folder_url_or_id:", folder_url_or_id)
        raise ValueError("❌ Could not extract folder ID from URL.")
    
    

def download_all_from_folder(folder_id_or_url, save_path="Data/Documents"):
    folder_id = extract_folder_id(folder_id_or_url)
    drive = authenticate_gdrive()

    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    
    if not file_list:
        print("❌ No files found in the folder.")
        return []

    print(f"✅ Found {len(file_list)} files in the folder.")

    # Create save_path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    downloaded_paths = []

    for file in file_list:
        file_name = file['title']
        file_path = os.path.join(save_path, file_name)
        try:
            file.GetContentFile(file_path)
            print(f"✅ Downloaded: {file_name}")
            
            #add files to the list of downloaded paths
            downloaded_paths.append(file_path) 
        except Exception as e:
            print(f"❌ Failed to download {file_name}: {e}")
            return []
    return downloaded_paths
    
