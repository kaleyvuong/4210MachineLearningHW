#-------------------------------------------------------------------------
# AUTHOR: Kaley Vuong
# FILENAME: decision_tree_2.py
# SPECIFICATION: Calculates accuracy of 3 datasets each trained 10 times
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 Day
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['4210HW2/contact_lens_training_1.csv', '4210HW2/contact_lens_training_2.csv', '4210HW2/contact_lens_training_3.csv']

features_dict = {
    "Young" : 1,
    "Prepresbyopic" : 2,
    "Presbyopic" : 3,

    "Myope" : 1,
    "Hypermetrope" : 2,

    "Yes" : 1,
    "No" : 2,

    "Normal" : 1,
    "Reduced" : 2
}

classes_dict = {
    "Yes" : 1,
    "No" : 2
}

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append(row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    for row in dbTraining:
        X.append([features_dict[row[0]], features_dict[row[1]], features_dict[row[2]], features_dict[row[3]]])

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

        Y.append([classes_dict[row[4]]])

    total_accuracy = 0
    #Loop your training and test tasks 10 times here
    for i in range (10):

       #Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)
        clf = clf.fit(X, Y)

       #Read the test data and add this data to dbTest
       #--> add your Python code here
        #Reading the training data in a csv file
        dbTest = []

        with open('4210HW2/contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append(row)
        
        correct_predict = 0
        total_predict = len(dbTest)

        for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            test_features = [[features_dict[data[0]], features_dict[data[1]], features_dict[data[2]], features_dict[data[3]]]]

            class_predicted = clf.predict(test_features)[0]

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here

            true_label = classes_dict[data[4]]
            if class_predicted == true_label:
                correct_predict += 1
        
        accuracy = correct_predict / total_predict
        total_accuracy += accuracy
            
    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    avg_accuracy = total_accuracy / 10 

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"Accuracy trained on {ds} : {avg_accuracy}")
