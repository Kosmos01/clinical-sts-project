import matplotlib.pyplot as plt
import numpy as np


def plotDefaults(listOfPearsons):
    #algorithms = ('Support\nVector\nClassifier', 'Random\nForest', 'K-Nearest\nNeighbors', 'Gaussian\nNaive\nBayes',
    #              'Logistic\nRegression')

    algorithms = ('Linear\nRegression','Ridge\nRegression','Bayesian\nRegression','Ada Boost\nRegressor','Gradient\nBoosting','Averaged\nPredictions')

    barWidth = 0.25

    r1 = np.arange(len(algorithms))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    #r4 = [x + barWidth for x in r3]

    label1 = 'Feature Set 1 (p >= 0.60)'
    label2 = 'Feature Set 2 (p >= 0.70)'
    label3 = 'Feature Set 3 (All)'

    plt.figure(num=None, figsize=(9.5, 3), dpi=950, facecolor='w', edgecolor='k')

    lineWidth = 3.0

    '''
    plt.plot(r1, listOfPearsons[0], color='crimson', label=label1, linewidth=lineWidth)
    plt.plot(r2, listOfPearsons[1], color='dodgerblue', label=label2, lineWidth=lineWidth)
    plt.plot(r3, listOfPearsons[2], color='forestgreen', label=label3, lineWidth=lineWidth)
    '''

    bar1 = plt.bar(r1, listOfPearsons[0], color='crimson', width=barWidth, edgecolor='white',label=label1)
    bar2 = plt.bar(r2, listOfPearsons[1], color='dodgerblue', width=barWidth, edgecolor='white',label=label2)
    bar3 = plt.bar(r3, listOfPearsons[2], color='forestgreen', width=barWidth, edgecolor='white',label=label3)
    #plt.xticks([r + (barWidth+0.05) for r in range(len(listOfPearsons[0]))],algorithms)

    plt.axes().set_ylim([0.50,0.80])
    plt.axes().set_yticks(np.arange(0.50,0.80,0.02))
    plt.axes().tick_params('both', labelsize=14)
    plt.xticks([r + (barWidth) for r in range(len(listOfPearsons[0]))], algorithms)
    plt.axes().set_ylim([0.50, 1.0])
    plt.axes().set_yticks(np.arange(0.50, 1.01, 0.25))
    # plt.ylabel('Pearson\ncorrelation\ncoefficient',rotation=0,labelpad=105,size=24)
    # plt.xlabel('Machine Learning Algorithms',labelpad=20,size=24)
    # plt.title("Median Pearson correlation coefficients per Feature Set (Default Parameters)",size=24,fontweight='bold',y=1.06)
    # plt.legend(prop={'size':14})
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=2, fancybox=True, shadow=True, prop={'size': 12})

    for rect in bar1 + bar2 + bar3:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, '{0:.02g}'.format(height), ha='center', va='bottom',fontsize=8)

    #plt.gca().invert_yaxis()

    plt.savefig('cTAKES-Analysis-1.png', bbox_inches="tight")
    plt.show()


def plotTable(listOfPearsons):
    plt.figure(num=None, figsize=(9.5, 3), dpi=950, facecolor='w', edgecolor='k')
    col_labels = ['Linear','Ridge','Bayesian','Ada','Gradient','Averaging']
    table_vals = [listOfPearsons[0],listOfPearsons[1],listOfPearsons[2],listOfPearsons[3]]
    plt.table(cellText=table_vals,colLabels=col_labels,loc="center")
    plt.show()
    plt.savefig('cTAKES.Table.png',bbox_inches="tight")
    exit(0)


def plotTunes(tunes, pearsonsList, modelType):
    '''
    for pearsons in pearsonsList:
        if dataset == 0:
            plt.plot(tunes,pearsons,color='blue',linestyle='solid',linewidth=3,marker='o',markerfacecolor='blue',markersize=12,label='All features')
        elif dataset == 1:
            plt.plot(tunes,pearsons,color='red',linestyle='solid',linewidth=3,marker='o',markerfacecolor='red',markersize=12,label='Feature set 1')
        elif dataset == 2:
            plt.plot(tunes,pearsons,color='green',linestyle='solid',linewidth=3,marker='o',markerfacecolor='green',markersize=12,label='Feature set 2')

        dataset+=1
    '''
    plt.plot(tunes, pearsonsList, color='g', linestyle='solid', linewidth=3, marker='o', markerfacecolor='g',
             markersize=12)

    plt.axes().tick_params('both', labelsize=22)
    if modelType == 'SVC':
        # plt.title('Tuning Cost Values for Support Vector Classifier (Feature Set 2)',size=24,fontweight='bold',y=1.06)
        plt.xlabel('Cost Values', size=22, labelpad=15)

    elif modelType == 'RandomForest':
        # plt.title('Tuning Estimators for Random Forest (Feature Set 2)',size=24,fontweight='bold',y=1.06)
        plt.xlabel('Number of Estimators', size=22, labelpad=15)

    elif modelType == 'KNN':
        # plt.title('Tuning K for K-Nearest Neighbor (Feature Set 2)',size=24,fontweight='bold',y=1.06)
        plt.xlabel('K Values', size=22, labelpad=15)

    for i, v in enumerate(pearsonsList):
        plt.text(v, i, " " + '{0:.3g}'.format(v), color='black', va='center', fontweight='bold', size=15)

    plt.ylabel('Pearson\ncorrelation\ncoefficient', rotation=0, labelpad=90, size=22)
    plt.show()


def plotFinal(pearsons):
    algorithms = ('Support\nVector\nClassifier', 'Random\nForest', 'K-Nearest\nNeighbors', 'Gaussian\nNaive\nBayes',
                  'Logistic\nRegression')

    plt.figure(num=None, figsize=(3, 7), dpi=950, facecolor='w', edgecolor='k')
    plt.tick_params('both', labelsize=14)
    plt.barh(algorithms, pearsons, color="dodgerblue", align='center', alpha=0.5)
    plt.yticks(ticks=np.arange(len(algorithms)), labels=algorithms)
    # plt.xticks(np.arange(0,1,.50))
    # plt.axes().set_ylim([0.0, 0.8])
    plt.xlim(0, 1.0)
    plt.xticks(ticks=[0.0, 0.5, 1.0])
    # plt.axes().set_yticks(np.arange(0.0,1, 0.50))

    for i, v in enumerate(pearsons):
        plt.text(v, i, " " + str('{0:0.3g}').format(v), color='black', va='center', fontweight='bold', size=12)
    plt.gca().invert_yaxis()
    plt.savefig('sample.png', bbox_inches="tight")
    plt.show()
    '''
    y_pos = np.arange(len(algorithms))

    plt.figure(num=None, figsize=(7,3), dpi=500 ,facecolor='w', edgecolor='k')
    plt.bar(y_pos, pearsons, color='dodgerblue',width=0.40, edgecolor='white')




    #plt.xlabel('Machine Learning Models',size=22,labelpad=10)
    plt.xticks([r for r in range(len(pearsons))], algorithms)

    plt.axes().set_ylim([0.65, 0.80])
    plt.axes().set_yticks(np.arange(0.65, 0.80, 0.01))
    plt.axes().tick_params('y',labelsize=16)
    #plt.ylabel('Pearson\ncorrelation\ncoefficient', rotation=0, labelpad=90,size=22)
    #plt.title('Final Pearson correlation coefficients (Feature Set 2 + Tuned Parameters)',fontweight='bold',size=24,y=1.06)
    plt.savefig('sample.png', bbox_inches="tight")
    plt.show()
    '''


def plotFeatureSelection(pearsons, parameters):
    # 3, 7
    plt.figure(num=None, figsize=(3, 7), dpi=700, facecolor='w', edgecolor='k')
    plt.tick_params('both', labelsize=14)
    plt.barh(parameters, pearsons, color="saddlebrown", align='center', alpha=0.5)

    params = ['CID-Cosine', 'CWID-Cosine', 'POS-Cosine', 'NVA-Cosine', 'Sentence-Cosine', 'CID-Jaccard',
                  'CWID-Jaccard', 'POS-Jaccard', 'NVA-Jaccard', 'Sentence-Jaccard', 'W2V-Pubmed-Cosine', 'GloVe-Wiki-Cosine',
                  'W2V-Google-Cosine', 'NVA-Pubmed-Cosine', 'NVA-Wiki-Cosine', 'NVA-Google-Cosine', 'Sentence-W2V-Pubmed-WMD',
                  'Sentence-W2V-Google-WMD','NVA-W2V-Pubmed-WMD', 'NVA-W2V-Google-WMD','BioBERT-Cosine','ClinicalBERT-Cosine','DischargeBERT-Cosine','GeneralBERT-Cosine']

    plt.yticks(ticks=np.arange(len(params)), labels=params)
    # plt.xticks(np.arange(0,1,.50))
    # plt.axes().set_ylim([0.0, 0.8])
    plt.xlim(0, 1)
    plt.xticks(ticks=[0.0, 0.5, 1.0])
    # plt.axes().set_yticks(np.arange(0.0,1, 0.50))

    # for i, v in enumerate(pearsons):
    #    plt.text(v, i, " " + str('{0:0.3g}').format(v), color='black', va='center', fontweight='bold',size=12)
    plt.gca().invert_yaxis()
    plt.savefig('CLAMP.FeatureSelection.png', bbox_inches="tight")
    plt.show()
    '''
    plt.figure(num=None, figsize=(9.5,3), dpi=950 ,facecolor='w', edgecolor='k')
    fig, ax = plt.subplots()

    #width = 0.75  # the width of the bars
    ind = np.arange(len(pearsons))  # the x locations for the groups
    plt.subplots().ax().barh(parameters, pearsons, color="cornflowerblue",align='center',alpha=0.5)
    #ax.set_yticks(ind + width / 2)
    plt.subplots().ax().set_yticklabels(parameters, minor=False)
    #plt.xlabel('Pearson correlation coefficient',labelpad=15,size=22)

    for i, v in enumerate(pearsons):
        plt.text(v, i, " " + str('{0:0.3g}').format(v), color='black', va='center', fontweight='bold')
    ax.tick_params(labelsize=20)
    plt.gca().invert_yaxis()
    plt.savefig('sample.png', bbox_inches="tight")
    plt.show()
    '''
    '''

    y_pos = np.arange(len(parameters))

    plt.barh(parameters, pearsons, align='center', alpha=0.5,color='k')
    #plt.xticks(pearsons,rotation='vertical')

    plt.axes().set_xlim([0.0, 0.75])
    plt.axes().set_xticks(np.arange(0.0,0.75,0.05))
    plt.axes().tick_params(labelsize=22)
    plt.xlabel('Pearson correlation coefficient', size=22,labelpad=10)
    #plt.yticks(np.arange(min(y_pos), max(y_pos), 1.5))



    plt.gca().invert_yaxis()
    plt.title('Feature Selection',fontweight='bold',size=24,y=1.04)
    plt.show()
    '''


def plotBERTS(pearsons):
    parameters = ['bioBERT', 'clinicalBERT', 'dischargeBERT', 'maskedBERT', 'b + c',
     'b + d', 'b + m', 'c + d','c + m', 'd + m', 'b + c + d', 'b + c + m', 'b + d + m', 'c + d + m','b + c + d + m']

    print('Number of combinations: ' + str(len(parameters)))

    plt.tick_params('both', labelsize=14)
    plt.barh(parameters, pearsons, color="saddlebrown", align='center', alpha=0.5)
    plt.yticks(ticks=np.arange(len(parameters)), labels=parameters)
    # plt.xticks(np.arange(0,1,.50))
    # plt.axes().set_ylim([0.0, 0.8])
    plt.xlim(0, 1)
    plt.xticks(ticks=[0.0, 0.5, 1.0])
    # plt.axes().set_yticks(np.arange(0.0,1, 0.50))

    # for i, v in enumerate(pearsons):
    #    plt.text(v, i, " " + str('{0:0.3g}').format(v), color='black', va='center', fontweight='bold',size=12)
    plt.gca().invert_yaxis()
    #plt.savefig('sample.png', bbox_inches="tight")
    plt.show()


def plotLogistic(weights, parameters, classnumber):
    plt.figure(num=None, figsize=(8, 5), dpi=700, facecolor='w', edgecolor='k')

    lineWidth = 3.0

    plt.plot(parameters, weights, color='darkseagreen', linewidth=lineWidth)
    '''
    plt.plot(r1, listOfPearsons[0], color='blue', width=barWidth, edgecolor='white',label=label1)
    plt.plot(r2, listOfPearsons[1], color='red', width=barWidth, edgecolor='white',label=label2)
    plt.plot(r3, listOfPearsons[2], color='green', width=barWidth, edgecolor='white',label=label3)
    plt.plot(r4, listOfPearsons[3], color='k', width=barWidth, edgecolor='white',label=label4)
    plt.xlabel('Machine Learning Models',size=24)
    plt.xticks([r + (barWidth+0.05) for r in range(len(listOfPearsons[0]))],algorithms)

    plt.axes().set_ylim([0.50,0.80])
    plt.axes().set_yticks(np.arange(0.50,0.80,0.02))
    '''
    plt.axes().tick_params('both', labelsize=13)
    # plt.xticks([r + (barWidth + 0.05) for r in range(len(listOfPearsons[0]))], algorithms)
    plt.axes().set_ylim(-1.0, 1.0)
    plt.axes().axhline(lineWidth=2, linestyle='--')
    plt.xticks(rotation=60)
    plt.axes().set_yticks(np.arange(-1.0, 1.01, 0.5))
    for tick in plt.axes().xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
    # plt.axes().tick_params(axis='x', which='major', pad=15)
    # plt.ylabel('Pearson\ncorrelation\ncoefficient',rotation=0,labelpad=105,size=24)
    # plt.xlabel('Machine Learning Algorithms',labelpad=20,size=24)
    # plt.title("Median Pearson correlation coefficients per Feature Set (Default Parameters)",size=24,fontweight='bold',y=1.06)
    # plt.legend(prop={'size':14})
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #           ncol=2, fancybox=True, shadow=True, prop={'size': 12})
    figurename = str(classnumber) + '.png'
    if classnumber == 0:
        plt.savefig('class0.png', bbox_inches="tight")
    elif classnumber == 1:
        plt.savefig('class1.png', bbox_inches="tight")

    elif classnumber == 2:
        plt.savefig('class2.png', bbox_inches="tight")

    elif classnumber == 3:
        plt.savefig('class3.png', bbox_inches="tight")

    elif classnumber == 4:
        plt.savefig('class4.png', bbox_inches="tight")

    elif classnumber == 5:
        plt.savefig('class5.png', bbox_inches="tight")

    # plt.show()