# Final Project Python S7: Spambase Dataset

Chloé DEPERTHES, Victoria GAUTHIER, Rémi KALBE

## Introduction

Using the Spambase dataset, the project's objective is to construct a spam filter. 4601 emails, 57 characteristics, and a label are all included in this dataset (spam or not spam). The information may be found on the UCI Machine Learning Repository at https://archive.ics.uci.edu/ml/datasets/Spambase.

## Data

The dataset is composed of 4601 emails, 57 features and a label (spam or not spam). The features are the following:

- word_freq_WORD: percentage of words in the e-mail that match WORD, i.e. 100 \* (number of times the WORD appears in the e-mail) / total number of words in e-mail. A "word" in this case is any string of alphanumeric characters bounded by non-alphanumeric characters or end-of-string.
- char_freq_CHAR: percentage of characters in the e-mail that match CHAR, i.e. 100 \* (number of CHAR occurences) / total characters in e-mail
- capital_run_length_average: average length of uninterrupted sequences of capital letters
- capital_run_length_longest: length of longest uninterrupted sequence of capital letters
- capital_run_length_total: sum of length of uninterrupted sequences of capital letters = total number of capital letters in the e-mail
- spam: denotes whether the e-mail was considered spam (1) or not (0), i.e. unsolicited commercial e-mail.

## Data Preprocessing

### Data Cleaning

The dataset is already clean, there is no missing values.

### Data Splitting

A training set and a test set were created from the dataset. The test set is used to evaluate the model after it has been trained using the training set. To divide the dataset into two halves, we utilize the train test split method from sklearn. 40% of the dataset is used for the test set, while 60% is used for the training set.

### Data Scaling

To scale the data, we leverage the StandardScaler function from sklearn. We use the fit_transform function to fit the scaler to the training set and to transform the training set. We use the transform function to transform the test set.

## Model

We used the following models:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

## Evaluation

We used the following metrics to evaluate the models:

- Accuracy
- Confusion Matrix

## Conclusion

The best models are the following:

- XGBoost: 0.95
- Random Forest: 0.9521739130434783

## References

- https://archive.ics.uci.edu/ml/datasets/Spambase
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
- https://xgboost.readthedocs.io/en/latest/python/python_api.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
