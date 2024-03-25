import csv

def writeFileTsv(filePath: str):
    with open(filePath, "w") as tsfFile:
        tsvWriter = csv.writer(tsfFile, delimiter='\t')

        tsvWriter.writerow(["row1", "hello"])
        tsvWriter.writerow(["row2", "world"])

def demo():
    tsfFile = "./dummy.tsv"

    writeFileTsv(tsfFile)

demo()