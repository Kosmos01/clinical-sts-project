import xml.etree.ElementTree as ET
import os

def parseXML():
    currentDirectory = os.path.dirname(os.path.realpath(__file__))
    subDirectory = 'xmiFiles'
    fileType = '.xmi'
    prefix = 'sent'
    CUIDs = []
    root1CUI = set()
    root2CUI = set()

    writer = open('PairsConceptsScores.txt', 'a')

    for sentenceNumber in range(0, 3284):
        fileName = prefix + str(sentenceNumber + 1) + fileType
        filePath = os.path.join(currentDirectory, subDirectory, fileName)

        if (sentenceNumber + 1) % 2 == 0:
            root2 = ET.parse(filePath).getroot()

            for child in root1:
                if 'UmlsConcept' in child.tag:
                    root1CUI.add(child.attrib['cui'])

                if 'Sofa' in child.tag:
                    sent1 = child.attrib['sofaString']

            for child in root2:
                if 'UmlsConcept' in child.tag:
                    root2CUI.add(child.attrib['cui'])

                if 'Sofa' in child.tag:
                    sent2 = child.attrib['sofaString']

            rowNumber = (sentenceNumber + 1) / 2
            rowIter = 1
            with open('sentenceScores.txt', 'r') as r:
                for line in r:
                    if rowIter == rowNumber:
                        formattedWrite = sent1 + '\t' + sent2 + '\t' + line + \
                                         'sent1 umls Concepts: ' + str(root1CUI) + '\n' \
                                                                                    'sent2 umls Concepts: ' + str(
                            root2CUI) + '\n' \
                                         'symmetric difference: ' + str(
                            root1CUI.symmetric_difference(root2CUI)) + '\n' \
                                                                         'intersection: ' + str(
                            root1CUI.intersection(root2CUI)) + '\n\n'
                        writer.write(formattedWrite)
                        break

                    rowIter += 1

            # print('Sentence Pair:' + sent1 + '\t' + sent2)
            # print('root1 umls Concepts: ' + str(root1UMLS))
            # print('root2 umls Concepts: ' + str(root2UMLS))
            # print('symmetric_difference: ' + str(root1UMLS.symmetric_difference(root2UMLS)))
            root1CUI.clear()
            root2CUI.clear()

        else:
            root1 = ET.parse(filePath).getroot()


def main():
    parseXML()

if __name__ == '__main__':
    main()