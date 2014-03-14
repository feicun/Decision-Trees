**Construct a decision tree, and then using it to classify instances from several different datasets.**

Note: Calculating the precision, recall and accuracy for multi-class is a little slow, please be patient.

Validation Steps:
The major part which need to be modify is the main method and "isPositive" method (above main method) in decisionTree.py

1. If the data set has multi-class (like nursery.arff), just modify the file name in these lines:
    Line 361    filename = 'nursery.arff'

Then add this line to main method:
    calculateForMultiClass(trainingData, testingData, attributes, None)

Don't forget to delete:
    runEvaluation5times(data, attributes, None)
    runZeroREvaluation5times(data, attributes, None)

Then run decisionTree.py, the result will be displayed.


2. If the data set has two classes (like breast-cancer.arff, tennis.arff), please modify the file name in these lines:
    Line 361    filename = 'nursery.arff'

Then add these lines to main method:
    runEvaluation5times(data, attributes, None)
    runZeroREvaluation5times(data, attributes, None)

Don't forget to delete:
    calculateForMultiClass(trainingData, testingData, attributes, None)

Then modify this line:
    Line 358    return "no-recurrence-events" in result
The string in this line should be the value that represent positive value of the attribute. For example, in "tennis.arff", which is "no", and in "breast-cancer.arff", which is "no-recurrence-events"

Then run decisionTree.py, the result will be displayed.