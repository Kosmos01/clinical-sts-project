import tensorflow as tf
import re
import torch
from pytorch_pretrained_bert import BertConfig, BertForPreTraining
import os
import json
import pickle
import numpy as np

def convert_tf_checkpoint_to_pytorch():
    # gave error originally. Solution found at: https://github.com/tensorflow/models/issues/2676
    tf_path = 'weights/biobert_v1.1_pubmed/model.ckpt-1000000'
    init_vars = tf.train.list_variables(tf_path)
    excluded = ['BERTAdam', '_power', 'global_step']
    init_vars = list(filter(lambda x: all([True if e not in x[0] else False for e in excluded]), init_vars))
    print(init_vars)

    names = []
    arrays = []

    for name, shape in init_vars:
        print("Loading TF weights {} with shape {}".format(name,shape))
        array = tf.train.load_variable(tf_path,name)
        names.append(name)
        arrays.append(array)

    config = BertConfig.from_json_file('weights/biobert_v1.1_pubmed/bert_config.json')
    print('Building Pytorch model from configuration {}'.format(str(config)))
    model = BertForPreTraining(config)

    for name, array in zip(names,arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    print("Save PyTorch model to {}".format('weights/'))
    torch.save(model.state_dict(), 'weights/pytorch_weight')

def embeddings(command,whichBert):
    # extract_features.py is a BERT class

    if whichBert == 'bioTest':
        file = 'bioBertTest.json'
        zeros = np.zeros(768)
    elif whichBert == 'clinicalTest':
        file = 'clinicalBertTest.json'
        zeros = np.zeros(768)
    elif whichBert == 'dischargeTest':
        file = 'dischargeBertTest.json'
        zeros = np.zeros(768)
    elif whichBert == 'maskedTest':
        file = 'maskedUncasedBertTest.json'
        zeros = np.zeros(1024)
    elif whichBert == 'bio':
        file = 'bioBert.json'
        zeros = np.zeros(768)
    elif whichBert == 'clinical':
        file = 'clinicalBert.json'
        zeros = np.zeros(768)
    elif whichBert == 'discharge':
        file = 'dischargeBert.json'
        zeros = np.zeros(768)
    elif whichBert == 'masked':
        file = 'maskedUncasedBert.json'
        zeros = np.zeros(1024)
    else:
        print('something went wrong!')
        exit(0)

    os.system(command)

    #os.system('python3 ' + ' extract_features.py \
    #            --input_file=inputForBert.txt \
    #            --vocab_file=weights/biobert_v1.1_pubmed/vocab.txt \
    #            --bert_config_file=weights/biobert_v1.1_pubmed/bert_config.json \
    #            --init_checkpoint=weights/biobert_v1.1_pubmed/model.ckpt-1000000 \
    #            --output_file=output.json')

    embedding_vectors = []
    print("[progress] Processed records: ")
    with open(file) as f:
        # each line is a json which has embedding information of a single document/description
        for line in f:
            d = json.loads(line)
            # 768 is hidden size
            #doc_vector = np.zeros(768)
            doc_vector = zeros

            # starting from 1 to skip [CLS] and to -1 to skip [SEP]
            for i in range(1, len(d['features']) - 1):
                # feature vector of the current token
                feature_vector = np.array(d['features'][i]['layers'][0]['values'])

                # adding to the document vector
                doc_vector = doc_vector + feature_vector

            # -2 is for excluding [CLS] and [SEP] tokens
            number_of_tokens = len(d['features']) - 2

            # since we want to compute the average of vector representations of tokens
            doc_vector = np.divide(doc_vector, number_of_tokens)

            embedding_vectors.append(doc_vector)

    print("[log] Saving the embedding vectors into file...")
    # saving the embedding vectors in a pickle file
    with open("embedding_vector.pkl", "wb") as f:
        pickle.dump(embedding_vectors, f)

    n = len(embedding_vectors)
    dim = len(embedding_vectors[0])
    print("Number of vectors: " + str(n))
    print("Vector dimension: " + str(dim))
    return embedding_vectors

def loadEmbeddings(whichBert):
    # extract_features.py is a BERT class

    if whichBert == 'bio':
        file = 'bioBert.json'
        zeros = np.zeros(768)
    elif whichBert == 'clinical':
        file = 'clinicalBert.json'
        zeros = np.zeros(768)
    elif whichBert == 'discharge':
        file = 'dischargeBert.json'
        zeros = np.zeros(768)
    elif whichBert == 'masked':
        file = 'maskedUncasedBert.json'
        zeros = np.zeros(1024)
    else:
        print('something went wrong!')
        exit(0)

    embedding_vectors = []
    print("[progress] Processed records: ")
    with open(file) as f:
        # each line is a json which has embedding information of a single document/description
        for line in f:
            d = json.loads(line)
            # 768 is hidden size
            #doc_vector = np.zeros(768)
            doc_vector = zeros
            # starting from 1 to skip [CLS] and to -1 to skip [SEP]
            for i in range(1, len(d['features']) - 1):
                # feature vector of the current token
                feature_vector = np.array(d['features'][i]['layers'][0]['values'])

                # adding to the document vector
                doc_vector = doc_vector + feature_vector

            # -2 is for excluding [CLS] and [SEP] tokens
            number_of_tokens = len(d['features']) - 2

            # since we want to compute the average of vector representations of tokens
            doc_vector = np.divide(doc_vector, number_of_tokens)

            embedding_vectors.append(doc_vector)

    print("[log] Saving the embedding vectors into file...")
    # saving the embedding vectors in a pickle file
    with open("embedding_vector.pkl", "wb") as f:
        pickle.dump(embedding_vectors, f)

    n = len(embedding_vectors)
    dim = len(embedding_vectors[0])
    print("Number of vectors: " + str(n))
    print("Vector dimension: " + str(dim))
    return embedding_vectors

def parseForBert():
    print('Parsing data set...')

    sentences = []
    with open('clinicalSTS2019.test.txt') as r:
        for sentence in r:
            sentenceTokens = sentence.split('\t')
            sentences.append(sentenceTokens[0].strip())
            sentences.append(sentenceTokens[1].strip())

    print(len(sentences))
    with open('inputForBert.txt','a+') as a:
        for sentence in sentences:
            a.write(sentence + '\n')


def whichBERT(inputFile,vocabFile,bertConfig,checkpoint,outputFile):
    command = 'python3 extract_features.py --input_file=' + inputFile + \
              ' --vocab_file=' + vocabFile + \
              ' --bert_config_file=' + bertConfig + \
              ' --init_checkpoint=' + checkpoint + \
              ' --output_file=' + outputFile

    return command

def main():
    #convert_tf_checkpoint_to_pytorch()
    parseForBert()
    embeddings()


if __name__ == '__main__':
    main()