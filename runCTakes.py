import subprocess
import os
import time
import xml.dom.minidom



def parseRawDataSet():
    '''
    Since cTAKES creates an XMI file per document, this fuction takes in the raw dataset and puts each sentence
        in its own .txt file in the 'inputDirectory' where cTAKES will grab from and the scores per pair in a
        seperate .txt file in the current working directory
    :return:
    '''

    print('Parsing data set...')

    sentencePairs = []
    sentenceScores = []
    sent = 1
    with open('clinicalSTS2019.test.txt') as r:
        for sentence in r:
            sentenceTokens = sentence.split('\t')
            sentencePairs.append(sentenceTokens[0])
            sentencePairs.append(sentenceTokens[1])
            #sentenceScores.append(sentenceTokens[2])

    currentDirectory = os.path.dirname(os.path.realpath(__file__))
    subDirectory = 'inputDirectory'
    fileType = '.txt'

    for sentence in sentencePairs:
        fileName = str('sent' + str(sent)) + fileType
        filePath = os.path.join(currentDirectory, subDirectory, fileName)
        open(filePath, 'a').write(sentence)
        sent += 1

    #for scores in sentenceScores:
    #    open('sentenceScores.txt', 'a').write(scores)

    print('Done splitting sentences and scores (:')


def runCTakesPipeline():
    '''
    Runs the default clinical pipeline where it goes to the 'inputDirectory' and creates an xmi file per each sentence
    (.txt file) and is put into the 'outputDirectory'.
    :return:
    '''
    print('Running ctakes pipeline...')
    command = "apache-ctakes-4.0.0/bin/runClinicalPipeline.sh -i ~/PycharmProjects/SummerResearch/inputDirectory/ " \
              "--xmiOut ~/PycharmProjects/SummerResearch/outputDirectory/ --user amhatami --pass 'Zb3LJiXoT$Dj@TQ3'"
    try:
        subprocess.call(command, shell=True)
    except FileNotFoundError:
        print('validating user timed out.... rerunning command..')
        time.sleep(3.0)
        subprocess.call(command, shell=True)
    print('Done running the clinical pipeline (:')

def formatXMIFiles():
    '''
    After the default clinical pipeline is finished it outputs an xmi per each sentence where the xmi contents are on
    a single row. This function formats the xmi output into normal xml format to ease the inspection of the xmi file.
    This function grabs from the 'outputDirectory' and with the formatted xmi files puts them in 'xmiFiles'
    :return:
    '''
    print('Formatting XMI files...')
    currentDirectory = os.path.dirname(os.path.realpath(__file__))
    subDirectory = 'outputDirectory'
    newSubDirectory = 'xmiFiles'
    fileType = '.txt.xmi'
    newFileType = '.xmi'
    prefix = 'sent'
    for sentencePair in range(0, 3284):
        fileName = prefix + str(sentencePair + 1) + fileType
        filePath = os.path.join(currentDirectory, subDirectory, fileName)
        dom = xml.dom.minidom.parse(filePath)
        pretty_xml_as_string = dom.toprettyxml()

        fileName = prefix + str(sentencePair + 1) + newFileType
        filePath = os.path.join(currentDirectory, newSubDirectory, fileName)
        open(filePath, 'a').write(pretty_xml_as_string)

    print('Done converting to traditional xml format (:')