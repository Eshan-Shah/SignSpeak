import os
import shutil

# Path to the file and folder
file_path = 'data/gestures.csv'
folder_path = 'model'


def resetData():
    # Delete the file
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")
    else:
        print(f"File '{file_path}' not found.")


def resetModel():
    # Delete the folder and all its contents
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents have been deleted.")
    else:
        print(f"Folder '{folder_path}' not found.")


resetModel()