import sys, math, re
import cPickle as pickle
import readARFF
import random

### takes as input a list of class labels. Returns a float
### indicating the entropy in this data.
def entropy(data) :
    val_freq = {}
    entropy = 0.0

    for element in data :
        if (val_freq.has_key(element)):
            val_freq[element] += 1.0
        else:
            val_freq[element] = 1.0

    for freq in val_freq.values():
        entropy += (-freq/len(data)) * math.log(freq/len(data), 2)

    return entropy
    #you write thisk

### Compute remainder - this is the amount of entropy left in the data after
### we split on a particular attribute. Let's assume the input data is of
### the form:
###    [(value1, class1), (value2, class2), ..., (valuen, classn)]
def remainder(data) :
    possibleValues = set([item[0] for item in data])
    r = 0.0
    for value in possibleValues :
        c = [item[0] for item in data].count(value)  
        r += (float(c) / len(data) ) * entropy([item[1] for item in
                                                data if item[0] == value])
    return r


### selectAttribute: choose the index of the attribute in the current 
### dataset that minimizes the remainder. 
### data is in the form [[a1, a2, ..., c1], [b1,b2,...,c2], ... ]
### where the a's are attribute values and the c's are classifications.
### and attributes is a list [a1,a2,...,an] of corresponding attribute values
def selectAttribute(data, attributes) :
    index = 0
    particularAttrData = []
    particularAttrValue = () ###This one will be put in particularAttrData list
    miniRemainder = 100.0
    selectAttr = ""
    for attribute in attributes :
        particularAttrData = [] ### Clear the list every time
        for item in data :
            particularAttrValue = (item[index], item[-1])
            particularAttrData.append(particularAttrValue)
        index += 1

        newRemainder = remainder(particularAttrData)
        if newRemainder < miniRemainder :
            miniRemainder = newRemainder
            selectAttr = attribute

    return attributes.index(selectAttr)
    #you write this
    
### a TreeNode is an object that has either:
### 1. An attribute to be tested and a set of children; one for each possible 
### value of the attribute.
### 2. A value (if it is a leaf in a tree)
class TreeNode :
    def __init__(self, attribute, value) :
        self.attribute = attribute
        self.value = value
        self.children = {}

    def __repr__(self) :
        if self.attribute :
            return self.attribute
        else :
            return self.value

    ### a node with no children is a leaf
    def isLeaf(self) :
        return self.children == {}

    ### return the value for the given data
    ### the input will be:
    ### data - an object to classify - [v1, v2, ..., vn]
    ### attributes - the attribute dictionary
    ### classify()
    def classify(self, data, attributes) :
        if self.isLeaf():
            return self.value
        else:
            value = []
            for index in attributes :
                if attributes[index].keys()[0] == self.attribute :
                    value = attributes[index][attributes[index].keys()[0]] ##Possible values for current attributes
                    tempIndex = index
            for item in value :
                if item == data[tempIndex] :
                    leaf = self.children.get(item)
            return leaf.classify(data, attributes)
       #you write this

### a tree is simply a data structure composed of nodes (of type TreeNode).
### The root of the tree 
### is itself a node, so we don't need a separate 'Tree' class. We
### just need a function that takes in a dataset and our attribute dictionary,
### builds a tree, and returns the root node.
### makeTree is a recursive function. Our base case is that our
### dataset has entropy 0 - no further tests have to be made. There
### are two other degenerate base cases: when there is no more data to
### use, and when we have no data for a particular value. In this case
### we use either default value or majority value.
### The recursive step is to select the attribute that most increases
### the gain and split on that.


### assume: input looks like this:
### dataset: [[v1, v2, ..., vn, c1], [v1,v2, ..., c2] ... ]
### attributes: [a1,a2,...,an] }
def makeTree(dataset, alist, attributes, defaultValue) :
    if len(dataset) == 0:
        leafNode = TreeNode(None, defaultValue)
        leafNode.children = {}
        return leafNode

    elif len(alist) == 0 :
        leafNode = TreeNode(None, readARFF.computeZeroR(attributes, dataset))
        leafNode.children = {}
        return leafNode

    elif entropy([data[-1] for data in dataset]) == 0.0:
        leafNode = TreeNode(None, dataset[0][-1])
        leafNode.children = {}
        return leafNode

    else :
        selectAttr = selectAttribute(dataset, alist)
        rootAttr = TreeNode(alist[selectAttr], None)
        rootAttr.children = {}
        #possibleValues = []
        for tempIndex in range(len(attributes)) :
            if attributes[tempIndex].keys() == [alist[selectAttr]] :
                index = [tempIndex][0]
        possibleValues = attributes[index][alist[selectAttr]]

        defaultValue = readARFF.computeZeroR(attributes, dataset)
        for val in possibleValues :
            subSet = createSubSet(val, dataset, selectAttr)
            rootAttr.children.update({val: makeTree(subSet, alist[:selectAttr]+alist[selectAttr+1:], attributes, defaultValue)})

        return rootAttr
    # you write; See assignment & notes for description of algorithm


### Will return dataset without given selectAttribute
def createSubSet(value, dataset, selectAttribute) :
    subSet = []
    for data in dataset :
        if data[selectAttribute] == value :
            subSet.append(data[:selectAttribute]+data[selectAttribute+1:])

    return subSet


def createTrainAndTestData(data) :
    trainingData = []
    testingData = []
    inputDataAmount = int(len(data) * 0.8)
    random.shuffle(data)
    for i in range(inputDataAmount) :
        trainingData.append(data[i])

    for item in data :
        if item not in trainingData :
            testingData.append(item)

    return trainingData, testingData

def evaluation(trainingData, testingData, attributes, defaultValue) :
    #tp = 'yes'
    #tn = 'no'
    accuracy = 0.0
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    root = makeTree(trainingData, readARFF.getAttrList(attributes), attributes, defaultValue)

    for item in testingData :
        classValue = item[-1]
        result = root.classify(item[0:-1], attributes)

        if result == defaultValue:
            continue
        elif result == classValue:
            if isPositive(result):
                TP +=1
            else:
                TN +=1
        else:
            if isPositive(result):
                FP +=1
            else:
                FN +=1
    if TP == 0 :
        precision = 0
        recall = 0
    elif (TP + TN) == 0:
        accuracy = 0
    else :
        precision = float(TP)/float((TP + FP))
        recall = float(TP)/float((TP + FN))
        accuracy = float((TP + TN))/float((TP + FP + FN + TN))

    #print precision, recall, accuracy
    return precision, recall, accuracy


def runEvaluation5times(data, attributes, defaultValue) :
    avgPrecision = 0
    avgRecall = 0
    avgAccuracy = 0
    allPrecision = 0
    allRecall = 0
    allAccuracy = 0
    for i in range(5) :
        trainingData, testingData = createTrainAndTestData(data)
        currPrecision, currRecall, currAccuracy = evaluation(trainingData, testingData, attributes, defaultValue)
        #print currPrecision, currRecall, currAccuracy
        allPrecision += currPrecision
        allRecall += currRecall
        allAccuracy += currAccuracy

    avgPrecision = allPrecision / 5
    avgRecall = allRecall / 5
    avgAccuracy = allAccuracy / 5

    print "Run Normal Evaluation 5 times -----------------"
    print "Average Precision: ", avgPrecision
    print "Average Recall: ", avgRecall
    print "Average Accuracy: ", avgAccuracy
    print ""


def evaluationWithZeroR(trainingData, testingData, attributes, defaultValue) :
    accuracy = 0.0
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    #root = makeTree(trainingData, readARFF.getAttrList(attributes), attributes, defaultValue)
    result = readARFF.computeZeroR(attributes, testingData)
    for item in testingData :
        classValue = item[-1]
        #result = root.classify(item[0:-1], attributes)

        if result == defaultValue:
            continue
        elif result == classValue:
            if isPositive(result):
                TP +=1
            else:
                TN +=1
        else:
            if isPositive(result):
                FP +=1
            else:
                FN +=1
    if TP == 0 :
        precision = 0
        recall = 0
    elif (TP + TN) == 0:
        accuracy = 0
    else :
        precision = float(TP)/float((TP + FP))
        recall = float(TP)/float((TP + FN))
        accuracy = float((TP + TN))/float((TP + FP + FN + TN))

    #print precision, recall, accuracy
    return precision, recall, accuracy


### Run 5-fold cross validations with original ZeroR
def runZeroREvaluation5times(data, attributes, defaultValue) :
    avgPrecision = 0
    avgRecall = 0
    avgAccuracy = 0
    allPrecision = 0
    allRecall = 0
    allAccuracy = 0
    for i in range(5) :
        trainingData, testingData = createTrainAndTestData(data)
        currPrecision, currRecall, currAccuracy = evaluationWithZeroR(trainingData, testingData, attributes, defaultValue)
        allPrecision += currPrecision
        allRecall += currRecall
        allAccuracy += currAccuracy

    avgPrecision = allPrecision / 5
    avgRecall = allRecall / 5
    avgAccuracy = allAccuracy / 5

    print "Run ZeroR 5 times -----------------"
    print "Average Precision: ", avgPrecision
    print "Average Recall: ", avgRecall
    print "Average Accuracy: ", avgAccuracy


def calculateForMultiClass(trainingData, testingData, attributes, defaultValue) :
    alist = readARFF.getAttrList(attributes)
    root = makeTree(trainingData, alist, attributes, defaultValue)
    classes = set([data[-1] for data in trainingData])

    Precision = {}
    Recall = {}
    correctPredicted = {}
    timesPredicted = {}
    exampleLabeled = {}

    for item in classes :
        correctPredicted[item] = 0
        timesPredicted[item] = 0
        exampleLabeled[item] = 0

    for element in testingData :
        classValue = element[-1]
        result = root.classify(element[0:-1], attributes)
        #print result
        timesPredicted[result] += 1
        exampleLabeled[classValue] += 1

        if result == defaultValue:
            continue
        if result == classValue :
            correctPredicted[result] += 1

    allCorrect = 0
    for item in classes :
        #print item
        #print correctPredicted[item]
        if correctPredicted[item] == 0:
            Precision[item] = 0
            Recall[item] = 0
        else:
            Precision[item] = float(correctPredicted[item]) / float(timesPredicted[item])
            Recall[item] = float(correctPredicted[item]) / float(exampleLabeled[item])
        allCorrect += correctPredicted[item]

    accuracy = float(allCorrect) / len(testingData)

    print "For multi-class -----------"
    print "Precision: ", Precision
    print "Recall: ", Recall
    print "Accuracy: ", accuracy

### This part need be modified for different dataset
def isPositive(result) :
    return "no-recurrence-events" in result

if __name__ == '__main__' :
    filename = 'nursery.arff'

    attributes = readARFF.readArff(open(filename))[0]
    data = readARFF.readArff(open(filename))[1]
    alist = readARFF.getAttrList(attributes)
    trainingData, testingData = createTrainAndTestData(data)

    #runEvaluation5times(data, attributes, None)
    #runZeroREvaluation5times(data, attributes, None)

    calculateForMultiClass(trainingData, testingData, attributes, None)