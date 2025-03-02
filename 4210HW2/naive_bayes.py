#-------------------------------------------------------------------------
# AUTHOR: Kaley Vuong
# FILENAME: Naive Bayes
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 day
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

features_dict = {
   "Sunny" : 1, 
   "Overcast" : 2, 
   "Rain" : 3,

   "Hot" : 1,
   "Mild" : 2,
   "Cool": 3,

   "High" : 1,
   "Normal" : 2,
   
   "Weak" : 1,
   "Strong" : 2,

   "Yes" : 1,
   "No" : 2
}

#Reading the training data in a csv file
#--> add your Python code here
dbTraining = []
X = []
Y = []

with open('4210HW2/weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTraining.append(row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here

for row in dbTraining: 
  X.append([features_dict[row[1]], features_dict[row[2]], features_dict[row[3]], features_dict[row[4]]])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
  Y.append([features_dict[row[5]]]) 


#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
dbTest = []
with open('4210HW2/weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         dbTest.append(row)

#Printing the header os the solution
#--> add your Python code here
print("Day Outlook Temperature Humidity Wind PlayTennis Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for row in dbTest:
  test_sample = [features_dict[row[1]], features_dict[row[2]], features_dict[row[3]], features_dict[row[4]]]

  probabilities = clf.predict_proba([test_sample])[0]
  predicted_class = clf.predict([test_sample])[0]

  predicted_label = list(features_dict.keys())[list(features_dict.values()).index(predicted_class)]

  confidence = probabilities[predicted_class - 1]

  if confidence >= 0.75:
    print(f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {predicted_label} {confidence:.3f}")




