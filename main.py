from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import svm
import csv
from sklearn import ensemble
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import ctakesAttributes
import runCTakes
import plotting
import Berts
from itertools import combinations
import json
import string
from sklearn.model_selection import RandomizedSearchCV
import subprocess

def computePearson(gsScores,sysScores,gs='gs.txt',sys='sys.txt'):
    with open(gs,'w') as a:
        for score in gsScores:
            a.write(str(score)+'\n')

    with open(sys,'w') as a:
        for score in sysScores:
            a.write(str(round(score,2))+'\n')

    out = subprocess.Popen(['./correlation-noconfidence.pl', 'gs.txt', 'sys.txt'],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()

    print('Script Pearson: ' + str(stdout))

def finalTest(train,test,model,parameters):
    X_train, y_train = train[parameters], train['score']
    X_test = test[parameters]

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)


    trained_model = model.fit(X_train,y_train)
    predictions = trained_model.predict(X_test)

    return predictions

def writeFinalScores(predictions,txtfile):
    with open(txtfile,'w') as a:
        for score in predictions:
            if score > 5.0:
                score = 5.0

            if score < 0.0:
                score = 0

            a.write(str(round(score,2))+'\n')

def trainDevTest(fileName):
    df = pd.read_csv(fileName)

    #35
    df = df.sample(frac=1,random_state=35).reset_index(drop=True)

    # 1150 train, 164 for dev, 328 for test set
    train, dev, test = df[:1150], df[1150:1314], df[1314:]
    return train, dev, test

def featureSelection(model,train,dev,parameters,gsFile='gs.txt',sysFile='sys.txt',type='None'):
    #for feature selection
    #X_train, y_train = train.drop(parameters,axis=1), train['score']
    #X_dev, y_dev = dev.drop(parameters,axis=1), dev['score']
    # for bert comparison!
    #if type=='fs' and len(parameters) == 1:
    if type=='fs':
        X_train, y_train = train[parameters].values.reshape(-1,1), train['score']
        X_dev, y_dev = dev[parameters].values.reshape(-1,1), dev['score']
    else:
        X_train, y_train = train[parameters], train['score']
        X_dev, y_dev = dev[parameters], dev['score']

    # for defaults
    X_train = StandardScaler().fit_transform(X_train)
    X_dev = StandardScaler().fit_transform(X_dev)
    #y_train = np.round(y_train)
    #y_dev = np.round(y_dev)

    trainedModel = model.fit(X_train,y_train)
    predictions = trainedModel.predict(X_dev)
    #computePearson(y_dev,predictions,gs=gsFile,sys=sysFile)

    pearson = pearsonr(y_dev,predictions)[0]
    print(round(pearson,3))
    #return predictions
    return pearson

def otherRuns(model,train,dev,parameters,type='None'):
    if type=='fs':
        X_train, y_train = train[parameters].values.reshape(-1,1), train['score']
        X_dev, y_dev = dev[parameters].values.reshape(-1,1), dev['score']
    else:
        X_train, y_train = train[parameters], train['score']
        X_dev, y_dev = dev[parameters], dev['score']

    # for defaults
    X_train = StandardScaler().fit_transform(X_train)
    X_dev = StandardScaler().fit_transform(X_dev)

    trainedModel = model.fit(X_train,y_train)
    predictions = trainedModel.predict(X_dev)

    pearson = pearsonr(y_dev,predictions)[0]
    return pearson, predictions

def chooseModel(modelType,kernel='linear',cost=1,num_estimators=200,random_state=0,neighbors=5,depth=3,leafs=5):
    if modelType == 'GaussianNB':
        return GaussianNB()

    elif modelType == 'LogisticRegression':
        #return LogisticRegression(multi_class='multinomial', solver='sag', random_state=random_state,max_iter=400)
        #return Lasso(random_state=random_state,max_iter=400)
        #return linear_model.LinearRegression()
        #return linear_model.BayesianRidge()
        #return ensemble.ExtraTreesRegressor()
        #return ensemble.AdaBoostRegressor()
        #return tree.DecisionTreeRegressor()
        return svm.SVR()

    elif modelType == 'SVC':
        return SVC(kernel = kernel,C=cost,random_state=random_state)

    elif modelType == 'RandomForest':
        return RandomForestClassifier(n_estimators=num_estimators, random_state=random_state)

    elif modelType == 'SGDClassifier':
        return SGDClassifier(random_state=random_state)

    elif modelType == 'KNN':
        return KNeighborsClassifier(n_neighbors=neighbors)

    elif modelType == 'DecisionTree':
        return DecisionTreeClassifier(random_state=random_state,max_depth=depth,min_samples_leaf=leafs)

    else:
        print('Invalid model chosen!')
        exit(0)


def main():

    runCTakes.parseRawDataSet()
    runCTakes.runCTakesPipeline()
    runCTakes.formatXMIFiles()

    with open('inputForBert.txt','a+') as a:
        for sent in posWords:
            sent  = sent.strip()
            a.write(sent.strip(string.punctuation) + '\n')

    Berts.parseForBert()


    print('creating bio command')
    bioBert = Berts.whichBERT(inputFile='inputForBert.txt',
                      vocabFile='weights/biobert_v1.1_pubmed/vocab.txt',
                      bertConfig='weights/biobert_v1.1_pubmed/bert_config.json',
                      checkpoint='weights/biobert_v1.1_pubmed/model.ckpt-1000000',
                      outputFile='bioBertTest.json')

    print('creating clinical command')
    clinicalBert = Berts.whichBERT(inputFile='inputForBert.txt',
                      vocabFile='weights/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/vocab.txt',
                      bertConfig='weights/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/bert_config.json',
                      checkpoint='weights/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/model.ckpt-150000',
                      outputFile='clinicalBertTest.json')

    print('creating discharged command')
    dischargeBert = Berts.whichBERT(inputFile='inputForBert.txt',
                      vocabFile='weights/pretrained_bert_tf/biobert_pretrain_output_disch_100000/vocab.txt',
                      bertConfig='weights/pretrained_bert_tf/biobert_pretrain_output_disch_100000/bert_config.json',
                      checkpoint='weights/pretrained_bert_tf/biobert_pretrain_output_disch_100000/model.ckpt-100000',
                      outputFile='dischargeBertTest.json')

    print('creating masked command')
    maskedUncasedBert = Berts.whichBERT(inputFile='inputForBert.txt',
                      vocabFile='weights/wwm_uncased_L-24_H-1024_A-16/vocab.txt',
                      bertConfig='weights/wwm_uncased_L-24_H-1024_A-16/bert_config.json',
                      checkpoint='weights/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt',
                      outputFile='maskedUncasedBertTest.json')


    # test dataset
    print('working on bio embeddings')
    bioEmbeddings = Berts.embeddings(bioBert,whichBert='bioTest')
    print('working on clinical embeddings')
    clinicalEmbeddings = Berts.embeddings(clinicalBert,whichBert='clinicalTest')
    print('working on discharged embeddings')
    dischargeEmbeddings = Berts.embeddings(dischargeBert,whichBert='dischargeTest')
    print('working on masked embeddings')
    maskedEmbeddings = Berts.embeddings(maskedUncasedBert,whichBert='maskedTest')


    print('calculating attributes')
    ctakesAttributes.calculateAttributes(bio=bioEmbeddings,clinical=clinicalEmbeddings,
                                         discharged=dischargeEmbeddings,masked=maskedEmbeddings)

 
    ################################################################################
    ###### Below is the feature selection, plotting, tuning, and final run of models
    #################################################################################

    df1 = pd.read_csv('cTAKES.Train.csv')
    #df2 = pd.read_csv('CLAMP.Train.csv')

    df1 = df1.drop(['sent1', 'sent2'], axis=1)
    #df2 = df2.drop(['sent1', 'sent2','score'], axis=1)
    #df3 = pd.concat([df1,df2],axis=1)
    # 35
    #df3 = df3.sample(frac=1, random_state=35).reset_index(drop=True)

    #
    train, dev = df1[:1438], df1[1438:]

    ###############################################
    ####### BERT features (comparing only bert models)
    ###############################################
    '''
    # write out to files
    #train.to_csv('trainBerts.csv',index=False)
    #dev.to_csv('devBerts.csv',index=False)
    #test.to_csv('testBerts.csv',index=False)

    berts = ['bioBERT','clinicalBERT','dischargeBERT','maskedBERT']
    allCombinations = []
    for x in range(1,5):
        combs = combinations(berts,x)
        for comb in list(combs):
            allCombinations.append(list(comb))

    pearsons = []
    for combin in allCombinations:
        parametersToTest = []
        for comb in combin:
            parametersToTest.append(comb)

        pearsons.append(trainDev(model=linear_model.Ridge(),train=train,dev=test,parameters=parametersToTest,type='fs'))

    print(pearsons)
    plotting.plotBERTS(pearsons)
    '''
    ###########################################
    ###### FEATURE COMPARISON
    ###########################################
    parameters = ['CUI-Cosine', 'CUI-Text-Cosine', 'POS-Cosine', 'NVA-Cosine', 'Sentence-Cosine', 'CUI-Jaccard',
                  'CUI-Text-Jaccard', 'POS-Jaccard', 'NVA-Jaccard', 'Sentence-Jaccard', 'Pubmed-Cosine', 'Wiki-Cosine',
                  'Google-Cosine', 'NVA-Pubmed-Cosine', 'NVA-Wiki-Cosine', 'NVA-Google-Cosine', 'WMD-Pubmed',
                  'WMD-Google','NVA-WMD-Pubmed', 'NVA-WMD-Google','bioBERT','clinicalBERT','dischargeBERT','maskedBERT']

    parameters_clamp = ['CLAMP-CUI-Cosine','CLAMP-CUI-Text-Cosine', 'CLAMP-POS-Cosine', 'CLAMP-NVA-Cosine', 'CLAMP-Sentence-Cosine', 'CLAMP-CUI-Jaccard',
                  'CLAMP-CUI-Text-Jaccard', 'CLAMP-POS-Jaccard', 'CLAMP-NVA-Jaccard', 'CLAMP-Sentence-Jaccard', 'CLAMP-Pubmed-Cosine', 'CLAMP-Wiki-Cosine',
                  'CLAMP-Google-Cosine', 'CLAMP-NVA-Pubmed-Cosine', 'CLAMP-NVA-Wiki-Cosine', 'CLAMP-NVA-Google-Cosine', 'CLAMP-WMD-Pubmed',
                  'CLAMP-WMD-Google', 'CLAMP-NVA-WMD-Pubmed', 'CLAMP-NVA-WMD-Google', 'CLAMP-bioBERT', 'CLAMP-clinicalBERT', 'CLAMP-dischargeBERT',
                  'CLAMP-maskedBERT']

    #combined_parameters = parameters_clamp + parameters

    pearsons = []

    for parameter in parameters:
        pearsons.append(featureSelection(model=linear_model.LinearRegression(),train=train,dev=dev,parameters=parameter,type='fs')) # exclude type='fs' when combining

    plotting.plotFeatureSelection(pearsons,parameters)
    exit(0)

    index = 0
    FeatureSet60 = []
    FeatureSet70 = []

    for pearson in pearsons:

        if pearson >= 0.70:
            FeatureSet70.append(parameters[index])

        if pearson >= 0.60:
            FeatureSet60.append(parameters[index])

        index += 1

    '''
    ##################################
    ###### clinical and masked - only collecting those features with above a threshold correlation
    ##################################
    FeatureSet50.append('bioBERT')
    FeatureSet50.append('dischargeBERT')
    FeatureSet50.append('clinicalBERT')
    FeatureSet50.append('maskedBERT')
    #FeatureSet60.append('bioBERT')
    #FeatureSet60.append('dischargeBERT')
    #FeatureSet60.append('clinicalBERT')
    #FeatureSet60.append('maskedBERT')
    ##################################
    ##################################
    ###### discharge and masked
    ###################################
    #FeatureSet50.append('dischargeBERT')
    #FeatureSet50.append('maskedBERT')
    #FeatureSet60.append('dischargeBERT')
    #FeatureSet60.append('maskedBERT')
    ####################################
    print(FeatureSet50)

    '''
    #print(FeatureSet60)

    listOfParams = []

    # full dataset

    #parameters = ['CUI-Cosine', 'CUI-Text-Cosine', 'POS-Cosine', 'NVA-Cosine', 'Sentence-Cosine', 'CUI-Jaccard', 'CUI-Text-Jaccard',
    #              'POS-Jaccard', 'NVA-Jaccard', 'Sentence-Jaccard', 'Pubmed-Cosine', 'Wiki-Cosine',
    #              'Google-Cosine', 'NVA-Pubmed-Cosine', 'NVA-Wiki-Cosine', 'NVA-Google-Cosine', 'WMD-Pubmed', 'WMD-Google',
    #              'NVA-WMD-Pubmed', 'NVA-WMD-Google', 'bioBERT', 'clinicalBERT', 'dischargeBERT']
    '''
    parameters = ['CUI-Cosine', 'CUI-Text-Cosine', 'POS-Cosine', 'NVA-Cosine', 'Sentence-Cosine',
                  'POS-Jaccard', 'NVA-Jaccard', 'Sentence-Jaccard', 'Pubmed-Cosine',
                  'Google-Cosine', 'NVA-Pubmed-Cosine', 'NVA-Wiki-Cosine', 'NVA-Google-Cosine', 'WMD-Pubmed',
                  'NVA-WMD-Pubmed', 'NVA-WMD-Google','bioBERT','clinicalBERT','dischargeBERT']
    '''


    listOfParams.append(FeatureSet60.copy())
    listOfParams.append(FeatureSet70.copy())
    listOfParams.append(parameters.copy())


    #listOfParams.append(FeatureSet60.copy())
    #listOfParams.append(FeatureSet70.copy())
    #listOfParams.append(parameters.copy())

    #listOfParams.append(combined_parameters.copy())

    pearsons = []


    ##################################################################
    ######## tuning regression models with randomSearch [n2c2 program]
    ##################################################################

    # linear
    linearParams = {
        'fit_intercept': [True,False],
        'normalize' : [True,False]
    }

    clf_linear = RandomizedSearchCV(estimator=linear_model.LinearRegression(),param_distributions=linearParams,
                                    n_jobs=-1,n_iter=150, cv=5,verbose=0,scoring='neg_mean_squared_error')

    # ridge
    ridgeParams = {
        'alpha': [3, 2, 1, 1e-02, 1e-04, 1e-06, 1e-08],
        'fit_intercept' : [True,False],
        'normalize' : [True, False],
        'solver' : ['auto','lsqr','sag'],
        'random_state': [42]
    }

    clf_ridge = RandomizedSearchCV(estimator=linear_model.Ridge(random_state=42), param_distributions=ridgeParams,
                             n_jobs=-1, n_iter=150, cv=5, verbose=0,scoring='neg_mean_squared_error')

    # bayesian
    bayesianParams = {
        'n_iter': [100, 200, 300, 400, 500, 600],
        'alpha_1': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03],
        'alpha_2': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03],
        'lambda_1': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03],
        'lambda_2': [1e-09, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03],
        'fit_intercept': [True,False],
        'normalize' : [True,False]
    }

    bayes_clf = RandomizedSearchCV(estimator=linear_model.BayesianRidge(), param_distributions=bayesianParams,
                             n_jobs=-1, n_iter=150, cv=5, verbose=0,scoring='neg_mean_squared_error')

    # ada boost regressor
    adaParams = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'learning_rate': [1.5, 1, 0.3, 0.1, 0.05, 0.01],
        'loss': ['linear', 'square', 'exponential'],
        'random_state' : [42]
    }

    ada_clf = RandomizedSearchCV(estimator=ensemble.AdaBoostRegressor(), param_distributions=adaParams,
                             n_jobs=-1, n_iter=150, cv=5, verbose=0,scoring='neg_mean_squared_error')

    # gradient boosting regressor
    gradParams = {
        'loss': ['ls', 'lad', 'huber', 'quantile'],
        'learning_rate': [1.5, 1, 0.3, 0.1, 0.05, 0.01],
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, ],
        'min_weight_fraction_leaf' : [0.0,0.1,0.2,0.3,0.4,0.5],
        'max_leaf_nodes' : [None,2,3,4],
        'max_depth' : [1,2,3,4,5,6],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'alpha': [0.9, 0.7, 0.5, 0.4, 0.5],
        'random_state' : [42]
    }

    grad_clf = RandomizedSearchCV(estimator=ensemble.GradientBoostingRegressor(),
                             param_distributions=gradParams,
                             n_jobs=-1, n_iter=150, cv=5, verbose=0,scoring='neg_mean_squared_error')


    train = train.append(dev, ignore_index=True)
    print('DEFAULT RUN:')
    for params in listOfParams:

        gatherPearsons = []

        # linear
        pear, pred = otherRuns(clf_linear,train,test,params)
        gatherPearsons.append(pear)
        predicts = pred

        # Ridge
        pear, pred = otherRuns(clf_ridge,train,test,params)
        gatherPearsons.append(pear)
        predicts += pred

        # bayesian Ridge
        pear, pred = otherRuns(bayes_clf,train,test,params)
        gatherPearsons.append(pear)
        predicts += pred

        # ada
        pear, pred = otherRuns(ada_clf,train,test,params)
        gatherPearsons.append(pear)
        predicts += pred

        # gradientBoost
        pear, pred = otherRuns(grad_clf,train,test,params)
        gatherPearsons.append(pear)
        predicts += pred

        predicts /= 5
        avg_predicts_pear = pearsonr(test['score'],predicts)[0]
        gatherPearsons.append(avg_predicts_pear)

        pearsons.append(gatherPearsons.copy())

        print("\n")
        print(params)
        print(clf_linear.best_params_)
        print(clf_ridge.best_params_)
        print(bayes_clf.best_params_)
        print(ada_clf.best_params_)
        print(grad_clf.best_params_)



    for p in pearsons:
        print(p)

    #print(pearsons)
    #for pearson in pearsons:
    #        print(np.median(pearson))
    #print(predictions)
    #predictions = predictions/5

    #print(predictions)

    #computePearson(dev['score'],predictions)

    plotting.plotDefaults(pearsons)
    exit(0)

    #######################################################################
    ################## TEST run with optimized parameters!!! [n2c2 program]
    #######################################################################
    train = train.append(dev, ignore_index=True)
    '''
    test1 = pd.read_csv('cTAKES.Test.csv')
    test2 = pd.read_csv('CLAMP.Test.csv')

    test1.drop(['sent1','sent2'],axis=1)
    test2.drop(['sent1','sent2'],axis=1)

    test = pd.concat([test1,test2],axis=1)
    '''
    test = pd.read_csv('cTAKES.Test.csv')
    test = test.drop(['sent1','sent2'],axis=1)

    #test1 = pd.read_csv('cTAKES.Test.csv')
    #test2 = pd.read_csv('CLAMP.Test.csv')

    #test = pd.concat([test1,test2],axis=1)
    #test = test.drop(['sent1','sent2'],axis=1)


    linear_final = linear_model.LinearRegression(normalize=True,fit_intercept=True)
    ridge_final = linear_model.Ridge(solver="sag",normalize=False,fit_intercept=True,alpha=2,random_state=42)
    bayes_final = linear_model.BayesianRidge(normalize=True,n_iter=100,lambda_2=1e-06,lambda_1=1e-09,
                                             fit_intercept=True,alpha_2=1e-08,alpha_1=0.001)
    ada_final = ensemble.AdaBoostRegressor(random_state=42,n_estimators=100,loss="exponential",learning_rate=0.05)
    grad_final = ensemble.GradientBoostingRegressor(random_state=42,n_estimators=300,min_weight_fraction_leaf=0.0,
                                                    min_samples_split=6,min_samples_leaf=5,max_leaf_nodes=4,
                                                    max_features="log2",max_depth=6,loss="ls",learning_rate=0.05,
                                                    alpha=0.5)



    final_pearsons = []



    catchPrediction = finalTest(train=train,test=test,model=linear_final,parameters=parameters)
    writeFinalScores(catchPrediction,'linear_predictions_cTAKES_Custom.txt')
    predictions = catchPrediction

    catchPrediction = finalTest(train=train,test=test,model=ridge_final,parameters=parameters)
    writeFinalScores(catchPrediction,'ridge_predictions_cTAKES_Custom.txt')
    predictions += catchPrediction

    catchPrediction = finalTest(train=train,test=test,model=bayes_final,parameters=parameters)
    writeFinalScores(catchPrediction,'bayes_predictions_cTAKES_Custom.txt')
    predictions += catchPrediction

    catchPrediction = finalTest(train=train,test=test,model=ada_final,parameters=parameters)
    writeFinalScores(catchPrediction,'ada_predictions_cTAKES_Custom.txt')
    predictions += catchPrediction

    catchPrediction = finalTest(train=train,test=test,model=grad_final,parameters=parameters)
    writeFinalScores(catchPrediction,'gradient_predictions_cTAKES_Custom.txt')
    predictions += catchPrediction

    predictions /= 5
    writeFinalScores(predictions,'averaged_predictions_cTAKES_Custom.txt')





    ##############################################################################
    ######## Below is the code used for the classification models [Summer Program]
    ######## and was commented out quickly as we had to have a quick turn around 
    ######## to utilize regression models and BERTS for the n2c2 program
    ###############################################################################


    ######################################################################################################
    ######## Dev Section - tuning parameters of classification models (very limited parameters were tuned)
    ######################################################################################################

    '''
    print('\nTuning Parameters\n')
    listOfParams = []
    #listOfParams.append(FeatureSet35)
    listOfParams.append(FeatureSet50)
    svcCosts = [1,2,3,4,5,6,7,8,9,10]

    rfEstimators = [150,200,250,300,350,400,450,500,550,600]

    knnNeighbors = [1,2,3,4,5,6,7,8,9,10]

    for params in listOfParams:
        svcPearsons = []
        rfPearsons = []
        knnPearsons = []
    # run various costs/estimators
        for cost, estimators, neighbors in zip(svcCosts,rfEstimators,knnNeighbors):

            SVC = chooseModel(modelType='SVC',kernel='linear',cost=cost)
            rf = chooseModel(modelType='RandomForest',num_estimators=estimators)
            knn = chooseModel(modelType='KNN',neighbors=neighbors)

            svcPearsons.append(trainDev(SVC,train,dev,params))

            rfPearsons.append(trainDev(rf,train,dev,params))

            knnPearsons.append(trainDev(knn,train,dev,params))

        print(svcPearsons)
        print(max(svcPearsons))
        print(svcCosts[svcPearsons.index(max(svcPearsons))])


        print(rfPearsons)
        print(max(rfPearsons))
        print(rfEstimators[rfPearsons.index(max(rfPearsons))])

        print(knnPearsons)
        print(max(knnPearsons))
        print(knnNeighbors[knnPearsons.index(max(knnPearsons))])

        plotting.plotTunes(svcCosts,pearsonsList=svcPearsons,modelType='SVC')
        plotting.plotTunes(rfEstimators,pearsonsList=rfPearsons,modelType='RandomForest')
        plotting.plotTunes(knnNeighbors,pearsonsList=knnPearsons,modelType='KNN')
    '''
    ###################################################
    ####### TestSection With Classification Models [2019 Summer Research]
    ####################################################
'''

    knnModelName = 'KNN'

    # Logistic Regression info
    lrModelName = 'LogisticRegression'

    # Gaussian NB info
    gnbModelName = 'GaussianNB'

    # svc info
    svcModelName = 'SVC'

    # random forest info
    rfModelName = 'RandomForest'


    train.append(dev,ignore_index=True)

    print('TEST SECTION!!!\n')

    # 3 300 10
    # 3 250 7

    svcCost = 3
    rfEstimator = 250
    kValue = 7

    SVC = chooseModel(modelType='SVC',kernel='linear',cost=svcCost)
    rf = chooseModel(modelType='RandomForest',num_estimators=rfEstimator)
    knn = chooseModel(modelType=knnModelName,neighbors=kValue)
    gnb = chooseModel(modelType=gnbModelName)
    lr = chooseModel(modelType=lrModelName)

    pearsons = []

    print('\n' + str(svcModelName) + ': ')
    print('Cost: ' + str(svcCost))
    #pearsons.append(trainDev(SVC,train,test,FeatureSet50))

    print('\n' + str(rfModelName) + ': ')
    print('estimators: ' + str(rfEstimator))
    #pearsons.append(trainDev(rf, train,test,FeatureSet50))

    print('\n' + str(knnModelName) + ': ')
    print('neighbors: ' + str(kValue))
    #pearsons.append(trainDev(knn, train,test,FeatureSet50))

    # run Gaussian NB
    print('\n' + str(gnbModelName) + ': ')
    #pearsons.append(trainDev(gnb,train,test,FeatureSet50))

    # run logistic regression
    print('\n' + str(lrModelName) + ': ')
    pearsons.append(trainDev(lr,train,test,FeatureSet50))


    #print(pearsons)

    #plotting.plotFinal(pearsons)
'''
if __name__ == '__main__':
    main()