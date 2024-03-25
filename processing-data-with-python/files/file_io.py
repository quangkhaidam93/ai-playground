import shutil
import os

def copyFileToDir(filePath: str, directory: str):
    shutil.copy(filePath, directory)

def copyFile(fileSrc: str, fileDest: str):
    shutil.copyfile(fileSrc, fileDest)

def iterateInDir(directory: str):
    for root, dirs, files in os.walk(directory):
        print(root) # Path to current directory
        print(dirs) # Sub-folders in current directory
        print(files) # Files in current directory

def listDirectory(directory: str):
    # os.listdir -> list all files and directories in current directory
    filesAndFolders = os.listdir(directory)

    filesInDir = [file for file in filesAndFolders if os.path.isfile(os.path.join(directory + file))]
    foldersInDir = [folder for folder in filesAndFolders if os.path.isdir(os.path.join(directory + folder))]

    print(filesInDir)
    print(foldersInDir)

def getFileName(filePath: str):
    fileName = os.fsdecode(filePath)

    print(fileName)

def createNewDir(newDirectory: str):
    if not os.path.exists(newDirectory):
        os.makedirs(newDirectory)
