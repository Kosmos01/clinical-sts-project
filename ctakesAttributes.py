import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
import numpy as np
import gensim

def grabUMLSAttributes():
    '''
    grabCUIs -> goes to the directory labeled xmiFiles and iterates through each xmi file and extracts the
    concept IDs per each sentence (iteration).
    :return: a list of sublists where the index (plus 1) of the list corresponds to the sentence number
    '''
    currentDirectory = os.path.dirname(os.path.realpath(__file__))
    subDirectory = 'xmiFiles'
    fileType = '.xmi'
    prefix = 'sent'

    # variables we are going to extract and append to lists
    CUIs = []
    preferredText = []
    sentences = []
    posTags = []
    posWords = []
    tagsToGrab = ['NN','JJ','VB']


    for sentenceNumber in range(0, 824):

        fileName = prefix + str(sentenceNumber + 1) + fileType
        filePath = os.path.join(currentDirectory, subDirectory, fileName)

        sublistOfPosTags = []
        sublistOfPosWords = []
        #sublistOfCUIs = []
        #sublistOfPrefText = []

        setOfCUIs = set()
        setOfPrefTexts = set()
        root = ET.parse(filePath).getroot()

        for child in root:

            if 'ConllDependencyNode' in child.tag:

                try:
                    if any(tag in child.attrib['cpostag'] for tag in tagsToGrab):
                        sublistOfPosTags.append(child.attrib['cpostag'])
                        sublistOfPosWords.append(child.attrib['form'].lower())

                except:
                    x=0

            if 'UmlsConcept' in child.tag:
                #sublistOfCUIs.append(child.attrib['cui'].lower())
                #sublistOfPrefText.append(child.attrib['preferredText'].lower())

                setOfCUIs.add(child.attrib['cui'].lower())
                setOfPrefTexts.add(child.attrib['preferredText'].lower())

            if 'Sofa' in child.tag:
                sentences.append(child.attrib['sofaString'].replace('"','').lower())



        CUIs.append(' '.join(list(setOfCUIs)))
        preferredText.append(' '.join(list(setOfPrefTexts)))

        #CUIs.append(' '.join(sublistOfCUIs))
        #preferredText.append(' '.join(sublistOfPrefText))

        posTags.append(' '.join(sublistOfPosTags))
        posWords.append(' '.join(sublistOfPosWords))

    #print(len(CUIs))
    return CUIs, preferredText, sentences, posTags, posWords


def Tfidf(corpus):

    if isinstance(corpus[0],str):
        count_vectorizer = TfidfVectorizer(stop_words='english',lowercase=False)
    else:
        count_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)

    vectors = count_vectorizer.fit_transform(corpus)
    #print(counts)
    #print(count_vectorizer.vocabulary_)
    #print(count_vectorizer.idf_)

    return vectors


def cosSim(vectors):

    similarities = []

    cosineMatrix = cosine_similarity(vectors,vectors)
    #print(cosineMatrix.shape)

    for index in range(3284):
        if index % 2 == 0:
            similarities.append(cosineMatrix[index][index+1])

    #print(len(similarities))
    #print(similarities)

    return similarities

def jaccardSimilarity(tokenizedLists):

    jaccardSimilarities = []

    for index in range(3284):

        if index % 2 != 0:
            B = set(tokenizedLists[index])

            try:
                similarity = (len(A.intersection(B))/len(A.union(B)))
            except ZeroDivisionError:
                if len(A.intersection(B)) != 0:
                    jaccardSimilarities.append(0.33)
                else:
                    jaccardSimilarities.append(1.0)

            jaccardSimilarities.append(similarity)
        else:
            A = set(tokenizedLists[index])

    return jaccardSimilarities

def wordMovers(embeddingModel, list):
    wmd = []

    for index in range(3284):

        if index % 2 != 0:
            doc2 = list[index]
            wmd.append(embeddingModel.wmdistance(doc1,doc2))
        else:
            doc1 = list[index]

    return wmd

def writeScoresToFile(features):
    indexIter = 0
    with open('clinicalSTS2019.train.txt','r') as r:
        with open('Test1.csv',mode='a') as file:
            writer = csv.writer(file, delimiter=',',quoting=csv.QUOTE_MINIMAL)
            for line in r:
                if indexIter == 0:
                    writer.writerow(['sent1','sent2','score','CUI-Cosine','CUI-Text-Cosine','POS-Cosine','NVA-Cosine','Sentence-Cosine','CUI-Jaccard',
                  'CUI-Text-Jaccard','POS-Jaccard','NVA-Jaccard','Sentence-Jaccard','Pubmed-Cosine','Wiki-Cosine',
                  'Google-Cosine','NVA-Pubmed-Cosine','NVA-Wiki-Cosine','NVA-Google-Cosine','WMD-Pubmed','WMD-Google',
                  'NVA-WMD-Pubmed','NVA-WMD-Google','bioBERT','clinicalBERT','dischargeBERT','maskedBERT'])

                writeLine = formatLine(trainingLine=line,features=features,index=indexIter)
                writer.writerow(writeLine)
                indexIter+=1

def formatLine(trainingLine, features, index):
    pairAndScore = trainingLine.rstrip()
    sentenceTokens = pairAndScore.split('\t')

    for x in range(len(features)):
        sentenceTokens.append(str(features[x][index]))

    return sentenceTokens


def loadModel(modelFile):
    '''
    input the path of the desired file (must be a .txt)
    Note: cannot read in the common crawl (glove) and google (w2v) since they are too large and do not fit in mem.
    :param modelFile:
    :return word embedding model:
    '''
    print('Loading ' + modelFile + '...')
    if '.bin' in modelFile:
        print('Loading W2V Model')
        return gensim.models.KeyedVectors.load_word2vec_format(modelFile,binary=True)
    elif '.txt' in modelFile:
        print("Loading Glove Model")
        f = open(modelFile, 'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.", len(model), " words loaded!")
        return model

def average_embeddings(model,processedSentences):
    captureEmebeddings = []

    for sentence in processedSentences:
        captureEmebeddings.append(get_mean_vector(model,sentence))

    return captureEmebeddings


def get_mean_vector(model, words):
    # remove out-of-vocabulary words
    if isinstance(model,dict):
        words = [word for word in words if word in model.keys()]
        if len(words) >= 1:

            return np.mean([model[x] for x in words],axis=0)
        else:
            return []
    else:
        words = [word for word in RegexpTokenizer(r'\w+').tokenize(words) if word in model.vocab]
        print(words)
        if len(words) >= 1:
            print(np.mean(model[words]))
            return np.mean(model[words], axis=0)
        else:
            return []

def preProcessCorpus(corpus):

    # importing basic stops
    basicStopsList = stopwords.words('english')

    # read in medical stop words
    stopSet = set([stop.strip() for stop in open('stopwords.txt','r').readlines()])

    # fill stopSet with basic stops from nltk
    for basicStop in basicStopsList:
        stopSet.add(basicStop)


    tokenizer = RegexpTokenizer(r'\w+')
    preProcessedCorpus = []

    for document in corpus:
        tokenizedDoc = tokenizer.tokenize(document)
        cleanDoc = []

        for token in tokenizedDoc:
            if token not in stopSet and token.lower() not in stopSet:
                cleanDoc.append(token.lower())

        preProcessedCorpus.append(cleanDoc)

    return preProcessedCorpus


def calculateAttributes(bio,clinical,discharged,masked):
    ## grab attributes
    print('Collecting attributes (:\n')
    CUIs, preferredText, sentences, posTags, posWords = grabUMLSAttributes()

    ## Calculate Tf-Idf of each attribute and perform cosine similarities
    print('Calculating tfidf-cossim (:\n')
    cuiTfIdfSim = cosSim(Tfidf(CUIs))
    preferredTextTfIdfSim = cosSim(Tfidf(preferredText))
    posTagsTfIdfSim = cosSim(Tfidf(posTags))
    posWordsTfIdfSim = cosSim(Tfidf(posWords))
    sentenceTfIdfSim = cosSim(Tfidf(sentences))

    ## Calculate jaccard similarities
    print('Calculating Jaccard sim (:\n')
    cuiJaccardSim = jaccardSimilarity(preProcessCorpus(CUIs))
    preferredTextJaccardSim = jaccardSimilarity(preProcessCorpus(preferredText))
    posTagsJaccardSim = jaccardSimilarity(preProcessCorpus(posTags))
    posWordsJaccardSim = jaccardSimilarity(preProcessCorpus(posWords))
    sentencesJaccardSim = jaccardSimilarity(preProcessCorpus(sentences))

    ## load in the desired model
    print('Loading in models (:\n')
    modelPubmed_w2v = loadModel('word2vecModels/pubmed.w2v/pubmed_s100w10_min.bin')
    modelWiki_glove = loadModel('gloveModels/glove.wikipedia.2014.Gigaword.5/glove.6B.300d.txt')
    modelGoogle_w2v = loadModel('word2vecModels/google.w2v/GoogleNews-vectors-negative300.bin')

    ## generate average word embeddings for the raw sentences
    print('averaging word embedings from sentences (:\n')
    averagePubmedEmebeddings = average_embeddings(modelPubmed_w2v, sentences)
    averageWikiEmbeddings = average_embeddings(modelWiki_glove, sentences)
    averageGoogleEmbeddings = average_embeddings(modelGoogle_w2v, sentences)

    ## generate average word embeddings for the pos word sentences
    print('averaging word embeddings from pos words (:\n')
    averagePosPubmedEmbeddings = average_embeddings(modelPubmed_w2v, posWords)
    averagePosWikiEmbeddings = average_embeddings(modelWiki_glove, posWords)
    averagePosGoogleEmbeddings = average_embeddings(modelGoogle_w2v, posWords)

    ## cosSim of average word embeddings
    print('calculating cos sim of avg embeddings from sentence (:\n')
    avgPubmedEmbdSim = cosSim(averagePubmedEmebeddings)
    avgWikiEmbdSim = cosSim(averageWikiEmbeddings)
    avgGoogleEmbdSim = cosSim(averageGoogleEmbeddings)

    ## cosSim of average POS words embeddings
    print('calculating cos sim of avg embeddings from pos words (:\n')
    avgWordPubmedEmbdSim = cosSim(averagePosPubmedEmbeddings)
    avgWordWikiEmbdSim = cosSim(averagePosWikiEmbeddings)
    avgWordGoogleEmbdSim = cosSim(averagePosGoogleEmbeddings)

    ## wordMoversDistance of sentences
    print('calculating word movers distances of sentences (:\n')
    sentenceWmdPubmed = wordMovers(modelPubmed_w2v, preProcessCorpus(sentences))
    sentenceWmdGoogle = wordMovers(modelGoogle_w2v, preProcessCorpus(sentences))

    ## wordMoversDistance of POS words
    print('calculating word movers distances of pos words (:\n')
    posWordsWmdPubmed = wordMovers(modelPubmed_w2v, preProcessCorpus(posWords))
    posWordsWmdGoogle = wordMovers(modelGoogle_w2v, preProcessCorpus(posWords))

    bioSim = cosSim(bio)
    clinicalSim = cosSim(clinical)
    dischargeSim = cosSim(discharged)
    maskedSim = cosSim(masked)


    features = [cuiTfIdfSim, preferredTextTfIdfSim, posTagsTfIdfSim, posWordsTfIdfSim, sentenceTfIdfSim, cuiJaccardSim,
                preferredTextJaccardSim, posTagsJaccardSim, posWordsJaccardSim, sentencesJaccardSim, avgPubmedEmbdSim,
                avgWikiEmbdSim,
                avgGoogleEmbdSim, avgWordPubmedEmbdSim, avgWordWikiEmbdSim, avgWordGoogleEmbdSim, sentenceWmdPubmed,
                sentenceWmdGoogle, posWordsWmdPubmed, posWordsWmdGoogle,bioSim,clinicalSim,dischargeSim,maskedSim]

    for x in range(len(features)):
        print(features[x][0])

    print('writing features (:\n')
    writeScoresToFile(features)
