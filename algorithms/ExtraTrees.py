from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import sys
sys.path.append('..')
from utils.utils import accuracy_roughly

# Random Forest Classifier
def Extra_Trees(X_train, X_test, y_train, y_test):
    extra_tree = ExtraTreesClassifier(criterion='entropy', max_depth=32, bootstrap=True, random_state=42, n_estimators=256, min_samples_split=2)
    extra_tree.fit(X_train, y_train)
    y_pred = extra_tree.predict(X_test)

    # accuracy
    print("ExtraTreesClassifier")
    print("accuracy:", accuracy_score(y_pred, y_test)*100,"%")
    print()
    print("accuracy(roughly):", accuracy_roughly(y_pred, y_test)*100,"%")
    print()
    print("confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("classificaion report")
    print(classification_report(y_test, y_pred))