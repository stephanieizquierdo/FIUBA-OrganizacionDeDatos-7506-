import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn import preprocessing

def show_roc_curve(model, model_name, X_test, X_train, y_test, y_train):
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, model.predict_proba(X_train)[:, 1])

    zero_test = np.argmin(np.abs(thresholds_test))
    zero_train = np.argmin(np.abs(thresholds_train))

    plt.plot(fpr_test, tpr_test, label = "ROC Curve " + model_name + " Test")
    plt.plot(fpr_train, tpr_train, label = "ROC Curve  " + model_name + " Train")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr_test[zero_test], tpr_test[zero_test], 'o', markersize=10, label="threshold zero test",
             fillstyle = "none", c ="k", mew = 2)
    plt.plot(fpr_train[zero_train], tpr_train[zero_train], 'x', markersize=10, label="threshold zero train",
             fillstyle = "none", c = "k", mew = 2)
    plt.legend(loc = 4)
    plt.show()


def show_confusion_matrix(y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(dpi=120)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', yticklabels=["Bajo valor", "Alto valor "],
                xticklabels=["Bajo valor", "Alto valor"], ax=ax)
    ax.set_title("Matriz de confusion")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")


def show_roc_auc_score(model, model_name, X_test, y_test):
    roc_auc_score_result = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("AUC-ROC score para " + model_name + ": {:.3f}".format(roc_auc_score_result))


def show_metrics(modelo, y_pred, y_test, X_test, nombre):
    print('Accuracy: ' + str(accuracy_score(y_pred,y_test)))
    print('----------------------------------------------')
    print('Precision: ' + str(precision_score(y_pred,y_test)))
    print('----------------------------------------------')
    print('Recall: ' + str(recall_score(y_pred, y_test)))
    print('----------------------------------------------')
    print('F1_score: ' + str(f1_score(y_pred, y_test)))
    print('----------------------------------------------')
    show_roc_auc_score(modelo, nombre, X_test, y_test)


def show_cv_metrics(cv_result):
    print('Accuracy: ' + str(cv_result["test_accuracy"].mean()))
    print('----------------------------------------------')
    print('Precision: ' + str(cv_result["test_precision"].mean()))
    print('----------------------------------------------')
    print('Recall: ' + str(cv_result["test_recall"].mean()))
    print('----------------------------------------------')
    print('Auc Roc score: ' + str(cv_result["test_roc_auc"].mean()))
    print('----------------------------------------------')


def write_predictions(predictions : np.array, model_name, user_ids):
    file = open("PrediccionesHoldout/" + model_name + ".csv", "w")
    file.write("id,tiene_alto_valor_adquisitivo\n")
    i = 0
    for prediction in predictions:
        file.write(str(user_ids[i]) + "," + str(prediction) + "\n")
        i = i + 1
    file.close()
