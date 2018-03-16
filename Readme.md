10-fold cross validation used for evaluation. Average accuracy was used as evaluation metric 

Parameter experiments  

1. Decision tree: clf = DecisionTreeClassifier()
Accuracy: 97.6%

2. Perceptron: clf = Perceptron(max_iter=1000)  
Accuracy: 94.8%
3. Perceptron: clf = Perceptron(eta0=0.5, max_iter=1000)  
Accuracy: 96.8%
4. Perceptron: clf = Perceptron(eta0=0.5, penalty='l1', alpha=0.001, max_iter=1000)   
Accuracy: 94.0%

5. Neural network: clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1) 
Accuracy: 97.6%
6. Neural network: clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2))
Accuracy: 84.8%
7. Neural network: clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2, 3), random_state=1)
Accuracy: 98.0%

8. Deep learning: clf = learn.DNNClassifier(hidden_units=[10, 20, 10],
                              feature_columns=learn.infer_real_valued_columns_from_input(X),
                              optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05))
Accuracy: 98.0%
9. Deep learning: clf = learn.DNNClassifier(hidden_units=[10, 5, 10, 3],
                              feature_columns=learn.infer_real_valued_columns_from_input(X),
                              optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05))
Accuracy: 98.0%
10. Deep learning: clf = learn.DNNClassifier(hidden_units=[10, 5, 4, 3, 6, 8, 4],
                              feature_columns=learn.infer_real_valued_columns_from_input(X),
                              optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05))
Accuracy: 90.0%

11. SVM: clf = SVC()
Accuracy: 97.1%
12. SVM: clf = SVC(kernel='poly', C=2)
Accuracy: 98.4%
13. SVM: clf = SVC(kernel='poly', C=0.5)
Accuracy: 97.6%

14. Naive Bayes: clf = GaussianNB()
Accuracy: 94.8%

15. Logistic regression: clf = LogisticRegression()
Accuracy: 97.3%
16. Logistic regression: clf = LogisticRegression(solver='liblinear', C=2)
Accuracy: 97.2%
17. Logistic regression: clf = LogisticRegression(solver='liblinear', C=2, penalty='l1')
Accuracy: 97.6%

18. KNN: clf = KNeighborsClassifier(n_neighbors=5)
Accuracy: 97.2%
19. KNN: clf = KNeighborsClassifier(n_neighbors=3)
Accuracy: 98.0% 
20. KNN: clf = KNeighborsClassifier(n_neighbors=7)
Accuracy: 96.8%

21. 