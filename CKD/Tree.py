import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
from six import StringIO  
import pydot
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

Train = False

# Model accuracy function
def accuracy(pred, y, showwrong = False):
    classification = pred == y
    correct = sum(classification)
    total = len(classification)

    acc = 100 * (correct / total)
    print("accuracy is:", acc)

    if showwrong == True:
        y_ones = sum(y)
        y_zeros = len(y) - y_ones
        indices = [i for i, x in enumerate(classification) if not x]
        wrong_pred = [x[1] for x in enumerate(pred) if x[0] in indices]
        zeros = sum(wrong_pred)
        ones = len(wrong_pred) - zeros
        print(100 * (zeros/y_zeros), "percent of zeros were incorrectly classified")
        print(100 * (ones/y_ones), "percent of ones were incorrectly classified")

# Creating a train-test split

path = "Chronic_Kidney_Dsease_data.csv"
df = pd.read_csv(path)

# Note we oversample the zero class due to it's significant underrepresentation

oversample = RandomOverSampler(sampling_strategy=0.5)

X = df.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis=1)
y = df['Diagnosis']
print(y.value_counts())

X, y = oversample.fit_resample(X, y)

y = y.to_list()

features = list(X.columns.values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train model
if Train == True:
    dtree = DecisionTreeClassifier(criterion="gini", max_depth = 4)
    dtree = dtree.fit(X_train, y_train)
else:
    filename = 'd_tree_model.sav'
    dtree = pickle.load(open(filename, 'rb'))

# Save model
filename = 'd_tree_model.sav'
pickle.dump(dtree, open(filename, 'wb'))

# Train and Test accuracies
pred = dtree.predict(X_train)
accuracy(pred, y_train)

pred = dtree.predict(X_test)
accuracy(pred, y_test, showwrong=True)


dot_data = StringIO() 
classes = ["No Kidney Disease", "Kidney Disease"]
tree.export_graphviz(dtree, feature_names=features, class_names=classes, filled=True, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph[0].write_pdf("tree.pdf")

importances = dtree.feature_importances_
indices = np.argsort(importances)[40:]
indices = np.flip(indices)
importances = importances[indices]
f = np.array([features[i] for i in indices])

data = np.c_[f, importances]

df = pd.DataFrame(data, columns=['Features', 'Relative Importances'])
df.to_csv('importances.csv', index = False) 
