import xml.etree.ElementTree as ET

def readFileXML(file: str):
    tree = ET.parse(file)

    root = tree.getroot()

    authors = [getTagText(authorTag) for authorTag in getTagChildrenByName(root, "author")]

    print(authors)

def getTagAttributes(element: ET.Element):
    return element.attrib

def getTagText(element: ET.Element):
    return element.text

def getTagChildren(element: ET.Element):
    return element.findall("*")

def getTagChildrenByName(element: ET.Element, name: str):
    return element.findall(name)

def demo():
    xml_file = "./dummy.xml"

    readFileXML(xml_file)

demo()

