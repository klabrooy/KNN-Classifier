#Kara La'Brooy 757553
#COMP30027 Project A: KNN Classifier

#This program has a driver function, does not take input from console,
#automatically outputs evaluation and runtime to console

import csv
import math
import random
import timeit
import operator

#constants
NUMBER_OF_ATTRIBUTES = 8;
RING_THRESHOLD = 11;

start_time = timeit.default_timer()

def preprocess_data(filename):
    #takes a file and returns a list of lists
    f = open(filename, "r")
    data_set = list(csv.reader(f, delimiter=','))

    #cLose File
    f.close()

    return data_set

def compare_instance(instance_one, instance_two, method):
    if method == "euclidean":
        #ignore gender attribute since nominal
        i = 1
        dist = 0

        while i < NUMBER_OF_ATTRIBUTES:
            #ignore evaluation if not a number
            #if instance_one[i].isdigit() and instance_two[i].isdigit():
            dist += math.pow(float(instance_one[i])-float(instance_two[i]), 2)
            i += 1

        return math.sqrt(dist)


def get_neighbours(instance, training_data_set, k, method):
    #return k most similar neighbours from the training set given a test instance
    scores = []
    nearest_neighbours = []

    i = 0
    for data_instance in training_data_set:
        score = compare_instance(instance, training_data_set[i], method)
        #add the instance and its calculated distance to a list of distances
        scores.append((training_data_set[i], score));
        i += 1

    #find nearest neighbours, with a low to high sort on the score key in
    #the list
    scores.sort(key=operator.itemgetter(1))

    #print(scores[1])
    #print("\n")

    #pick only the nearest neighbours for voting
    i = 0
    while i < k:
        nearest_neighbours.append(scores[i])
        i += 1

    return nearest_neighbours

def predict_class(neighbours, method):
    if method == "majority":
        #rings <= 10
        young = 0
        #rings >= 11
        old = 0

        age = ["young", "old"]
        i = 0
        for neighbour in neighbours:
        #check ring size of neighbours
            if int(neighbours[i][0][NUMBER_OF_ATTRIBUTES]) < RING_THRESHOLD:
                young+=1
            else:
                old+=1
            i += 1

            #return predicted class labels
            if young > old:
                return "young"
            elif young < old:
                return "old"
            else:
                return random.choice(age)

def evaluate(data_set, predictions, metric):
    if metric == "accuracy":
        #an accuracy performance metric
        i = 0
        correct = 0
        actual = []

        #bulid a list of correct class labels for test instances
        for instance in data_set:
            if int(data_set[i][NUMBER_OF_ATTRIBUTES]) < RING_THRESHOLD:
                actual.append("young")
            else:
                actual.append("old")
            i += 1

        i = 0
        for instance in predictions:
            if actual[i] == predictions[i]:
                correct += 1
            i+=1

        return (float(correct)/len(predictions))

    if metric == "precision":
            #precision evaluation metric
            TP = 0
            FP = 0
            i = 0
            actual = []

            #bulid a list of correct class labels for test instances
            for instance in data_set:
                if int(data_set[i][NUMBER_OF_ATTRIBUTES]) < RING_THRESHOLD:
                    actual.append("young")
                else:
                    actual.append("old")
                i += 1

            #count true positives (young and young) and true negatives (!young and !young)
            i = 0
            for instance in predictions:
                if actual[i] == "young" and predictions[i] == "young":
                    TP += 1
                elif actual[i] == "old" and predictions[i] == "old":
                    FP += 1
                i+=1

            #return precision
            return (float(TP)/(TP+FP))


#----- program driver-----
data_set = preprocess_data("abalone.data.txt")

#randomly split into training and test
random.shuffle(data_set)
#partition value decides portion of whole data_set to reserve as test
partition = 0.33
test_partition = int(round(partition * len(data_set)))

test_data = data_set[:test_partition]
training_data = data_set[test_partition:]

#a list for holding the machines prediction for each test data
predictions = []

i = 0
for instance in test_data:
    #find its k nearest neighbours
    nearest_neighbours = get_neighbours(test_data[i], training_data, 15, "euclidean")
    #make a prediction based on voting
    prediction = predict_class(nearest_neighbours, "majority")
    predictions.append(prediction);
    i += 1;

#summary
print("Accuracy (%): \t\t", evaluate(test_data, predictions, "accuracy")*100)
print("Precision (%): \t\t", evaluate(test_data, predictions, "precision")*100)
print("Execution Time (s): \t", timeit.default_timer() - start_time)
