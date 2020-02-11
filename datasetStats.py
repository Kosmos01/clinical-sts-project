import csv
import os
from nltk.tokenize import RegexpTokenizer


def pairStats():
    '''
    Since cTAKES creates an XMI file per document, this fuction takes in the raw dataset and puts each sentence
        in its own .txt file in the 'inputDirectory' where cTAKES will grab from and the scores per pair in a
        seperate .txt file in the current working directory
    :return:
    '''

    print('Parsing data set...')
    tokenizer = RegexpTokenizer(r'\w+')
    sentencePairs = []
    sentenceScores = []

    with open('clinicalSTS2019.train.txt') as r:
        for sentence in r:
            aggregate = []
            sentenceTokens = sentence.split('\t')
            aggregate.append(tokenizer.tokenize(sentenceTokens[0]))
            aggregate.append(tokenizer.tokenize(sentenceTokens[1]))
            sentencePairs.append(aggregate.copy())
            sentenceScores.append(sentenceTokens[2].strip())



    print(len(sentencePairs))
    print(len(sentenceScores))
    print(sentencePairs[0])
    print(sentenceScores[0])
    print(sentencePairs[15][0])
    print(len(sentencePairs[10][0]))

    totalNumberOfWords = 0
    sent1Greater = 0
    sent2Greater = 0
    bothGreater = 0

    for sentencePair in sentencePairs:
        totalNumberOfWords += len(sentencePair[0])
        totalNumberOfWords += len(sentencePair[1])

        if len(sentencePair[0]) >= 21.41:
            sent1Greater+=1

        if len(sentencePair[1]) >= 21.41:
            sent2Greater+=1

        if len(sentencePair[0]) >= 21.41 and len(sentencePair[1]) >= 21.41:
            bothGreater+=1


    print('Total number of words: ' + str(totalNumberOfWords))
    print('Average number of words per sentence: ' + str(totalNumberOfWords/3284))
    print('Number of sentence 1 sentence over or equal to the average: ' + str(sent1Greater))
    print('Number of sentence 2 sentence over or equal to the average: ' + str(sent2Greater))
    print('Total number of sentences above the average: ' + str(sent1Greater+sent2Greater))
    print('Total number of sentences above the average: ' + str(sent1Greater+sent2Greater))
    print('Total number of both sentences greater than average: ' + str(bothGreater))

def main():
    pairStats()







if __name__ == '__main__':
    main()