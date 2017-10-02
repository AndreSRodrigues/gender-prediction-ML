import numpy as np
from sklearn.metrics import accuracy_score

#import models
from sklearn import tree
from sklearn import gaussian_process
from sklearn import neural_network

#[altura, peso, tamanho do sapato]
data = [
    [180,80,43], [190,83,44], [160,68,37], [155,66,36], [170,71,37], [175,75,40],[180,80,43],
    [180,80,43], [200,100,46], [175,78,41],[181,85,42], [158,57,35], [161,58,35], [169,61,38],
    [185,87,43], [160,70,38]
]
labels = [
    ['male'], ['male'], ['female'], ['female'], ['female'], ['male'], ['male'],
    ['male'],['male'], ['male'], ['male'], ['female'], ['female'],['female'],
    ['male'],['male']
]

trainingData = []
trainingLabel = []

#Split data into training and test data
def splitData (_data, _labels, percent):

    #Zip two lists together into another list
    #Then randomize the zipped list
    #Replace the original lists with the randomized ones
    zipData = list(zip(_data, _labels))
    np.random.shuffle(zipData)
    _data = [i[0] for i in zipData]
    _labels = [i[1] for i in zipData]

    #for the number loops determined by the percentage
    #Popout the first element of the original list and add it to the training list
    nLoops = int(len(_data)*percent)

    for i in range(nLoops):
        trainingData.append(_data.pop(0))
        trainingLabel.append(_labels.pop(0))

    #Save agin the new lists and return them
    data = _data
    labels = _labels
    return _data, _labels, trainingData, trainingLabel

splitData (data,labels, 0.5)
#print (trainingData)

numTests = 20
treeResults = []
gaussianResults = []
neuralNetResults = []

for i in range(numTests):
    #Classifiers
    clf_tree = tree.DecisionTreeClassifier()
    clf_gaussian = gaussian_process.GaussianProcessClassifier()
    clf_neuralNet = neural_network.MLPClassifier(hidden_layer_sizes=(100,), alpha=0.0001, learning_rate_init=0.0001)

    #Training
    treeFit = clf_tree.fit(trainingData, trainingLabel)
    gaussianFit = clf_gaussian.fit(trainingData,trainingLabel)
    neuralNetFit = clf_neuralNet.fit(trainingData, trainingLabel)

    #Predictions and accuraccy
    prediction_tree = treeFit .predict(data)
    accuraccy_tree = accuracy_score(labels, prediction_tree) * 100 #in percentage
    treeResults.append(accuraccy_tree)

    prediction_gaussian = gaussianFit.predict(data)
    accuraccy_gaussian = accuracy_score(labels,prediction_gaussian) * 100
    gaussianResults.append(accuraccy_gaussian)

    prediction_neuralNet = neuralNetFit.predict(data)
    accuraccy_neuralNet = accuracy_score(labels,prediction_neuralNet) *100
    neuralNetResults.append(accuraccy_neuralNet)

    #Print results
    #print ('Decision tree acurracy ', accuraccy_tree)
    #print ('Gaussian tree acurracy ', accuraccy_gaussian)
    #print ('Neural Net acurracy ', accuraccy_neuralNet)

treeMean = np.mean(treeResults)
gaussianMean = np.mean(gaussianResults)
neuralNetMean = np.mean(neuralNetResults)

print ('Decision tree results ', treeMean)
print ('Gaussian tree results ', gaussianMean)
print ('Neural Net results ', neuralNetMean)