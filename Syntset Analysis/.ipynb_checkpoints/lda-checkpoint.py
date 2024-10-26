import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

def apply_lda(numeric_df, y, score):
    n_components = 1
    lda = LDA(n_components=n_components)

    # Remove train_test_split, we will use cross-validation directly on the entire dataset
    classifier = LogisticRegression(max_iter=1000)

    skf = StratifiedKFold(n_splits=5)
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    for train_index, test_index in skf.split(numeric_df, y):
        X_train_fold, X_test_fold = numeric_df.iloc[train_index], numeric_df.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        X_lda = lda.fit_transform(X_train_fold, y_train_fold)
        lda_df = pd.DataFrame(data=X_lda, columns=[f'LD{i+1}' for i in range(n_components)])
        classifier.fit(lda_df, y_train_fold)

        X_test_lda = lda.transform(X_test_fold)
        X_test_df =  pd.DataFrame(data=X_test_lda, columns=[f'LD{i+1}' for i in range(n_components)])
        y_pred_fold = classifier.predict(X_test_df)

        precision_scores.append(precision_score(y_test_fold, y_pred_fold, average='macro'))
        recall_scores.append(recall_score(y_test_fold, y_pred_fold, average='macro'))
        accuracy_scores.append(classifier.score(X_test_df, y_test_fold))
        different_locs = np.where(y_test_fold != y_pred_fold)[0]

        # Map these back to the original indices in lda_df
        original_different_locs = test_index[different_locs]

        print("Locations with different values in the original dataset:", original_different_locs)
        print("Scores with different values in the original dataset:", score[original_different_locs])

    print("Cross-validated Precision:", sum(precision_scores) / len(precision_scores))
    print("Cross-validated Recall:", sum(recall_scores) / len(recall_scores))
    print("Cross-validated Accuracy:", sum(accuracy_scores) / len(accuracy_scores))
