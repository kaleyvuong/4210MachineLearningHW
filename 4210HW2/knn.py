#-------------------------------------------------------------------------
# AUTHOR: Kaley Vuong
# FILENAME: knn.py
# SPECIFICATION: Calculates the error rate of the KNN class predictions
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 day
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('4210HW2/email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append(row)

num_errors = 0
total_instances = len(db)

#Loop your data to allow each instance to be your test set
for i, test_instance in enumerate(db):
    #Add the training features to the 2D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    Y = []
    for j, instance in enumerate(db):
        if j != i:
            features = [float(k) for k in instance[:-1]]
            X.append(features)
    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
            label = 1 if instance[-1] == "spam" else 0
            Y.append(label)
    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testsample = [float(k) for k in test_instance[:-1]] 

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here

    class_predicted = clf.predict([testsample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    true_label = 1 if test_instance[-1] == 'spam' else 0

    if class_predicted != true_label:
        num_errors += 1

error_rate = num_errors / total_instances

#Print the error rate
#--> add your Python code here
print(f"Error rate:{error_rate}")





