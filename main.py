import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score

# Read data and pre-process
df = pd.read_csv("/Users/Peiyang/Desktop/ML/Assignment5/programming/Qualitative_Bankruptcy.data.txt", names=['IR', 'MR', 'FF', 'CR', 'CO', 'OP', 'cls'])
df = shuffle(df) # Shuffle
category_map = {'P': -1, 'A': 0, 'N': 1}
df.IR = df.IR.map(category_map)
df.MR = df.MR.map(category_map)
df.FF = df.FF.map(category_map)
df.CR = df.CR.map(category_map)
df.CO = df.OP.map(category_map)
df.OP = df.OP.map(category_map)
df.cls = df.cls.map({'B': 0, 'NB': 1})

X = df.iloc[:, 0:5]
Y = df.iloc[:, 6]


# Decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
acc_DT = cross_val_score(clf, X, Y, cv=10).mean() * 100
print("Decision tree average accuracy: %.1f%%" % (acc_DT))

# Perceptron
from sklearn.linear_model import Perceptron
clf = Perceptron(eta0=0.5, max_iter=1000)
acc_perceptron = cross_val_score(clf, X, Y, cv=10).mean() * 100
print("Perceptron average accuracy: %.1f%%" % (acc_perceptron))

# Neural network
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2, 3), random_state=1)
acc_NN = cross_val_score(clf, X, Y, cv=10).mean() * 100
print("Neural net average accuracy: %.1f%%" % (acc_NN))

# Deep learning
from tensorflow.contrib import learn
import tensorflow as tf
# tf.learn does not support cross_val_score, need to run manually
print('Testing deep learning model, this may take some time...')
acc_DL = 0.0
kf = KFold(n_splits=10, shuffle=True)
for train_index, test_index in kf.split(df):
    # Split data
    train_data = df.iloc[train_index, 0:5]
    train_target = df.iloc[train_index, 6]
    test_data = df.iloc[test_index, 0:5]
    test_target = df.iloc[test_index, 6]

    tf.logging.set_verbosity(tf.logging.ERROR)
    clf = learn.DNNClassifier(hidden_units=[10, 5, 10, 3],
                              feature_columns=learn.infer_real_valued_columns_from_input(X),
                              optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05))
    clf.fit(train_data, train_target, batch_size=128, steps=500)
    acc_DL += accuracy_score(test_target, list(clf.predict(test_data)))
print("Deep learning accuracy: %.1f%%" % (acc_DL * 10))

# SVM
from sklearn.svm import SVC
clf = SVC(kernel='poly', C=2)
acc_SVM = cross_val_score(clf, X, Y, cv=10).mean() * 100
print("SVM accuracy: %.1f%%" % (acc_SVM))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
acc_NB = cross_val_score(clf, X, Y, cv=10).mean() * 100
print("Naive Bayes accuracy: %.1f%%" % (acc_NB))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='liblinear', C=2, penalty='l1')
acc_LR = cross_val_score(clf, X, Y, cv=10).mean() * 100
print("Logic regression accuracy: %.1f%%" % (acc_LR))

# KNN
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=7)
acc_kNN = cross_val_score(clf, X, Y, cv=10).mean() * 100
print("K-Nearest neighbor accuracy: %.1f%%" % (acc_kNN))

# Bagging
# print("Bagging accuracy: %.1f%%" % (acc_Bagging))

# Random forest
# print("Random forest accuracy: %.1f%%" % (acc_RF))

# Adaboost
# print("Adaboost accuracy: %.1f%%" % (acc_AdaBoost))

# Gradient boosting
# print("Gradient boosting accuracy: %.1f%%" % (acc_GB))