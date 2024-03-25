def readFileText(filePath: str):
    with open(filePath, "r") as textFile:
        content = textFile.read() # Read full content of file as string
        lines = content.split(",")

        imageUrls: list[str] = []

        for line in lines:
            # Chaining replace function overall is the best option: 
            # Reference: https://stackoverflow.com/questions/3411771/best-way-to-replace-multiple-characters-in-a-string
            url = line.replace('[', '').replace(']', '').replace("'", '').replace("\n", "") 

            imageUrls.append(url)

        print(f"result imageUrls:\n")
        printResult(imageUrls)

def printResult(imageUrls: list[str]):
    print("[")

    for url in imageUrls:
        print(f"\t{url}")

    print("]")


def demo():
    txtFile = "./image_urls.txt"

    readFileText(filePath=txtFile)

demo()


