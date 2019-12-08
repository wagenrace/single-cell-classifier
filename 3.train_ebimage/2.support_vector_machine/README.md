# Support Vector Machine

A Support Vector Machine (SVM) is classifying by fitting an hyperplane of d-1 dimensions as border between classes.
d = number of dimensions of the data

## Naive Attempt

Using [Support Vector Machine](https://scikit-learn.org/0.21/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) we trained an estimator on the [EBImage features](../2.process-data/README.md) of the training data.
Also we trained on [2000 eigenvalues](../2.process-data/README.md) of the training set.
Eigenvalues with SVM is proven to work on [face detection](https://www.kaggle.com/serkanpeldek/face-recognition-on-olivetti-dataset)

### Parameters

(these are the default parameters of sklearn version 0.21.3)
- C=1.0
- kernel=’rbf’
- degree=3
- gamma=’auto_deprecated’
- coef0=0.0
- shrinking=True
- probability=False
- tol=0.001
- cache_size=200
- class_weight=None
- verbose=False
- max_iter=-1
- decision_function_shape=’ovr’
- random_state=None

### Experiment setup

We run a script to determine model performance and model stability. 
Every time the script is run another results in added to the list. 
These results show what the performance is (the mean) and how stable it get there (the variance).
This can be used to compare different models with each other and conclude if they have different stability. 

### Results

The balanced f1-score on the validation set is [0.328 ± 0.00286 with 8 model initializations](results/all_scores.csv).
This is very low, but what is interesting is the [confusion matrix](results/0/confusion_matrix.png) (shown is one example initialization).
The predictions are dominated by 3 classes: dopaminereceptor, EGFR, ROCK.
When looking at the [most common classes](../2.process-data/results/target_counts.tsv) we find the same 3 classes.
When we look at the next 3 most common classes most common class (adrenoceptor, DNA_intercalation, AMPA) we see very few predictions. Almost all of these images are labeled as one of the 3 dominated classes.
There are a few classes (e.q. Ca2, Cdc25, eNos, rac1) are well predicted and seems to be easy cases.

### Future Steps

- [ ] Balance training data. The model seems to favor classes that are very dominate.  
- [ ] Parameter optimization. Try different parameters to find best solution
