"""
Header: This file holds the Feature Model class which is used for roc curves.
It takes a partition and a feature to create a dictionary of frequencies of a positive outcome (adopted)
for individual feature values. Then, in the classify function, it can take any example
and predict outcome from feature model and the example's feature value.
By: Mitali and Angie
"""
#imports
from Partition import *

################################################################################
# CLASSES
################################################################################

class FeatureModel:

    def __init__(self, partition, feature):
        """
        The contructor takes a partition (Partition of the training dataset) and
        a feature (string) which is the sole feature that will be used for
        predictions. It creates a dictionary of frequencies of positive outcomes (adopted)
        for each feature value.
        """
        self.feature = feature #initalizes feature of interest within class
        dictionary = {} #creates dictionary
        for i in partition.F[feature]: #iterates through partition
            total = 0 #total animals
            adopted = 0 #total adopted
            for animal in partition.data: #iterates through each animal in partition
                if animal.features[feature] == i: #if it has feature
                    total += 1 #add to total features
                    if animal.label == 1: #adopted
                        adopted += 1 # add to total adopted
            if total ==0:
                #if there is no positive outcomes for feature value to prevent from dividing by zero
                dictionary[i] = 0
            else:
                dictionary[i] = adopted/total #find probability of feature value and desired outcome
        self.dictionary = dictionary #initalizes dictrionary within class
        #print(dictionary)

    def classify(self, example, threshold):
        """
        This helper method classifies one example (Example from the test
        dataset) as 0 or 1 using the given threshold.
        """
        #looks at feature model to determine prob positive based on feature value in example
        positive_prob = self.dictionary[example.features[self.feature]]
        if positive_prob >= threshold: #compares probability to threshold to classify example
            return 1
        else:
            return 0



################################################################################
# MAIN
################################################################################

def main():

    pass

if __name__ == "__main__":
    main()
