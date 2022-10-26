
"""
Header: This file takes in data about an animal shelter in Austin Texas and looks at incoming animal's
outcome_subtype (underlying conditions i.e. rabies), animal_type,
sex_upon_outcome (gender + spayed or nuetered), age_upon_outcome, breed,color.
Then, it uses entropy and roc curve analysis on these features with the label of outcome,
as either adopted (1) or euthanised (0). The purpose of this is to determine which features
are the best indicators of an animal's outcome when going into shelters.

Authors: Mitali and Angie
"""

#imports
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
from FeatureModel import *
from collections import OrderedDict
from Partition import *



def main():
    """
    Main function reads in data from excel sheet, runs roc and entropy methods on
    created train and test data.
    """
    train_partition, train_data = file_reader("data/Animal_Center_cleaned.xlsx") #creates train data
    test_partition, test_data = file_reader("data/Animal_Center_test.xlsx") #creates test data
    #list of features
    list_features = ["outcome_subtype","animal_type", "sex_upon_outcome","age_upon_outcome","breed","color"]
    colors = ['b', 'g', 'r', 'c', 'm', 'y'] #colors for roc curve graph
    #ROC methods:
    #creates curve for each feature
    for i in range(len(list_features)):
        create_roc(train_partition, test_partition, list_features[i], colors[i])
    #creates graph
    plt.legend()
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC curve for Animal Shelter data set")
    plt.show()
    #plt.savefig("figures/roc_curve.pdf", format='pdf')
    #entropy methods:
    print("Best feature using entropy:")
    print(best_feature(train_data, train_partition.F)) #finds best feature
    print("Second best feature using entropy:")
    print(second_best_feature(train_data, train_partition.F)) #finds second best feature



def file_reader(filename):
    """
    reads file makes partition using data frame, pandas, and numpy.
    """
    data_frame = pd.read_excel(filename, header=0) # header on line 0
    #Used pandas to filter to adopted and euthanised animals
    labels = []
    #list of features
    list_features = ["outcome_subtype","sex_upon_outcome","animal_type", "age_upon_outcome","breed","color"]
    shelter_data1 = data_frame.loc[data_frame["outcome_type"] == "Adoption",list_features]
    #creates a list of the labels in order where adopted is "1" and put down is "0"
    for i in range(len(shelter_data1)):
        labels.append(1)
    shelter_data2 = data_frame.loc[data_frame["outcome_type"] == "Euthanasia",list_features]
    for i in range(len(shelter_data2)):
        labels.append(0)
    #creates a nested array of the examples
    frames = [shelter_data1,shelter_data2]
    shelter_data = pd.concat(frames) #1256, 911 adopted, 345 put down
    array_data = shelter_data.to_numpy()
    list_examples = []
    #creates a list of examples
    for i in range(len(shelter_data)):
        example_dict = {}
        feature_array = array_data[i]
        #creates examples using example class for each animal
        for j in range(len(list_features)):
            example_dict[list_features[j]] = feature_array[j]
        list_examples.append(Example(example_dict,labels[i]))
    #creates F partition
    F = {}
    for feature in list_features:
        options = []
        for example in list_examples:
            if example.features[feature] not in options:
                options.append(example.features[feature])
        F[feature] = options
    partition = Partition(list_examples,F) #creates a partition
    return partition, list_examples


def arrayMaker(title,data_frame):
    """
    Takes in adopted and euthanised list of animals and concatinates them into a
    single data frame.
    """
    adopted = data_frame.loc[data_frame["outcome_type"] == "Adoption",title]
    euthanasia = data_frame.loc[data_frame["outcome_type"] == "Euthanasia",title]
    frames = [adopted,euthanasia]
    df = pd.concat(frames)  #concatinates frames
    array_data = df.to_numpy()
    feature_vals = []
    for i in range(len(array_data)): #puts it in array format
        if array_data[i] not in feature_vals:
            feature_vals.append(array_data[i])
    return feature_vals


#entropy methods

def best_feature(data,F):

    """
    Compares information gains to determine best feature
    """
    info_gain = info_gain_map(data, F)
    best_key = ""
    temp_highest = 0
    for key in info_gain: #iterates through all features and compares info gains
        value = info_gain[key]
        #print(key)
        #print(info_gain[key])
        if value > temp_highest:
            best_key = key
            temp_highest = value
    return best_key

def second_best_feature(data,F):
    """
    Compares information gains to determine second best feature
    """
    info_gain = info_gain_map(data, F)
    best_key = ""
    second_best = ""
    temp_second = 0
    temp_highest = 0
    for key in info_gain: #iterates through all features to compare info gains
        value = info_gain[key]
        if value > temp_highest:
            temp_second = temp_highest
            second_best = best_key
            best_key = key
            temp_highest = value
        elif value > temp_second:
            second_best = key
            temp_second = value
    return second_best

def info_gain_map(data, F):
    """
    Creates map of information gains for each feature
    """
    info_gain = {}
    for feature_name in F:
        info_gain[feature_name] = feature_gain(feature_name, data)
    return info_gain

def probability_overall(data, label_choice):
    """
    Finds overall probability of a label
    """
    count = 0
    total = 0
    for example in data: #counts occurances
        if example.label == label_choice:
            count += 1
        total += 1
    result = (count/total)
    inverse = 1- result
    final = -(result) * math.log2(result) - (inverse)* math.log2(inverse) #uses log formula
    return final


def conditional_entropy(data, label, feature):
    """
    Finds conditional entropy for an individual feature
    """
    sum = 0 #total entropy
    map_totals = {} #total counts with feature
    map_label = {} #total counts with feature and label
    for example in data:
        #creates and adds to two maps
        if example.features[feature] in map_totals:
            #if in map totals, increment up
            map_totals[example.features[feature]] += 1
        else:
            #otherwise, add feature value to map total
            map_totals[example.features[feature]] = 1
            map_label[example.features[feature]] = 0
        if example.label == label:
            map_label[example.features[feature]] += 1
    #creates the probabilities and does the calculations from the maps above to find entropy
    for key in map_label:
        prob_label = map_label[key] / map_totals[key]
        inverse = 1- prob_label
        if inverse == 0: # to prevent log 0
            final = -(prob_label) * math.log2(prob_label)
        elif prob_label == 0:
            final = -(inverse)* math.log2(inverse)
        else:
            final = -(prob_label) * math.log2(prob_label) - (inverse)* math.log2(inverse)
        #sum keeps track of info gain for each feature value
        sum += final * (map_totals[key]/ len(data))#number of examples with feature value
    return sum


def feature_gain(feature, data):
    """
    Calculates feature gain for one feature using conditional entropy and overall
    """
    overall = probability_overall(data, 1)
    entropy = conditional_entropy(data, 1, feature)
    return (overall - entropy)

def create_roc(train_partition, test_partition, feature, color):
    """
    takes in training and test data to create a roc curve for an individual feature.
    uses test data to create Feature Model to later classify train data.
    """
    f_true = FeatureModel(train_partition, feature)  #creates FeatureModel
    thresholds = np.linspace(-0.0001,1.1,2000) #creates a large array of possible thresholds
    x_vals = []
    y_vals = []
    thresh_index = 0
    for t in thresholds: #iterates through thresholds to get different points for graph
        falsePos =0
        falseNeg = 0
        truePos =0
        trueNeg = 0
        #iterates through each test example to classify each example
        for i in test_partition.data:
            #uses classify in Feature Model to predict outcome for animal
            pred= f_true.classify(i, thresholds[thresh_index])
            if i.label == 0 and pred == 1:
                falsePos += 1
            if i.label == 1 and pred == 1:
                truePos += 1
            if i.label == 0 and pred == 0:
                trueNeg += 1
            if i.label == 1 and pred == 0:
                falseNeg += 1
        #calculates false and true positive rates for graph
        falsePosRate = falsePos/(falsePos+trueNeg)
        truePosRate = truePos/(truePos+falseNeg)
        x_vals.append(falsePosRate)
        y_vals.append(truePosRate)
        thresh_index +=1
    #creates a roc curve graph for the feature based on false and true positives
    #from every test example at different thresholds
    plt.plot(x_vals, y_vals, color=color, label=feature)





if __name__ == '__main__':
    main()
