import io
import base64
import numpy as np
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from enum import Enum
import matplotlib
matplotlib.use('Agg')


plt = matplotlib.pyplot


class LRMUsing(Enum):
    GridSearchCV = 1
    RFE = 2


class LogisticRegressionModel:
    def __init__(self, X_train, y_train, X_test, y_test, using: LRMUsing):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # convert y_train and y_test to int
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

        if using == LRMUsing.GridSearchCV:
            self.model = GridSearchCV(LogisticRegression(), [{
                'penalty': ['l2'],
                'C': np.logspace(-5, 5, 11),
                'dual': [False]
            }], cv=5, verbose=0)
            self.model.fit(self.X_train, self.y_train)
        elif using == LRMUsing.RFE:
            self.model = RFE(LogisticRegression(), step=1,
                             n_features_to_select=5)
            self.model = self.model.fit(self.X_train, self.y_train)
        else:
            raise ValueError("Invalid value for using")

    def predict(self):
        return self.model.predict(self.X_test)

    def score(self):
        return self.model.score(self.X_test, self.y_test)

    def confusion_matrix_base64(self):
        model_logreg_rfe_cm = confusion_matrix(self.y_test, self.predict())

        f, ax = plt.subplots(figsize=(5, 5))

        sns.heatmap(model_logreg_rfe_cm, annot=True, linewidths=0.5,
                    linecolor="yellow", fmt=".0f", ax=ax, cmap="YlOrBr")
        plt.xlabel("y_pred")
        plt.ylabel("y_true")

        plot_as_io_bytes = io.BytesIO()
        plt.savefig(plot_as_io_bytes, format='jpg')
        plot_as_io_bytes.seek(0)

        b64png = base64.b64encode(plot_as_io_bytes.read())

        # Clean up
        plot_as_io_bytes.close()
        plt.clf()

        return b64png
