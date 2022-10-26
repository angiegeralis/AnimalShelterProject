"""
This file holds the example and partition class, which are used to store and manipulate data
Refrenced from previous labs.
"""

class Example:

    def __init__(self, features, label):
        """Helper class (like a struct) that stores info about each example."""
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label # in {0, 1}

class Partition:

    def __init__(self, data, F):
        """Store information about a dataset"""
        # list of Examples
        self.data = data
        self.n = len(self.data) #length of data set

        # dictionary. key=feature name: value=list of possible values
        self.F = F
