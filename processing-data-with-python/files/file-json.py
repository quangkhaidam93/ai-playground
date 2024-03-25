import json

def readFileJson(filePath: str):
    with open(filePath, "r") as jsonFile:
        data = json.load(jsonFile)

        print(data)

def demo():
    jsonFile = "./dummy.json"

    readFileJson(jsonFile)

demo()