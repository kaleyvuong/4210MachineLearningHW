#-------------------------------------------------------------------------
# AUTHOR: Kaley Vuong
# FILENAME: decision_tree.py
# SPECIFICATION: Outputs decision tree with given data table
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2 days
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('4210HW1/contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
#--> add your Python code here
# X =
for row in db:
    features = []
    # Age
    if row[0] == "Young":
        features.append(1)
    elif row[0] == "Prepresbyopic":
        features.append(2)
    elif row[0] == "Presbyopic":
        features.append(3)
    # Spectacle Prescription
    if row[1] == "Myope":
        features.append(1)
    elif row[1] == "Hypermetrope":
        features.append(2)
    # Astigmatism
    if row[2] == "Yes":
        features.append(1)
    elif row[2] == "No":
        features.append(2)
    # Tear Production Rate
    if row[3] == "Normal":
        features.append(1)
    elif row[3] == "Reduced":
        features.append(2)
    X.append(features)

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
#--> add your Python code here
# Y =
    if row[4] == "Yes":
        Y.append(1)
    elif row[4] == "No":
        Y.append(2)

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()