#import libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, r2_score, f1_score

sns.set(style="white")

# Load the data
dataset = pd.read_csv('iris.csv')

#Feature Names
dataset.columns = [colname.strip(' (cm)').replace(" ", "_") for colname in dataset.columns.tolist()]
features_names = dataset.columns.tolist()[:4]

#Feature Engineering
dataset['sepal_length_width_ratio'] = dataset['sepal_length'] / dataset['sepal_width']
dataset['petal_length_width_ratio'] = dataset['petal_length'] / dataset['petal_width']

#Select Features
dataset = dataset[['sepal_length', 
             'sepal_width', 
             'petal_length', 
             'petal_width', 
             'sepal_length_width_ratio', 
             'petal_length_width_ratio', 
             'target']]

#Training
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
X_train = train_data.drop('target', axis=1).values.astype("float32")
y_train = train_data.loc[:, 'target'].values.astype("int32")

X_test = test_data.drop('target', axis=1).values.astype("float32")
y_test = test_data.loc[:, 'target'].values.astype("int32")

#Log Reg
logreg = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=100, multi_class='multinomial')
logreg.fit(X_train, y_train)
prediction_lr = logreg.predict(X_test)
cm = confusion_matrix(prediction_lr, y_test)
f1 = f1_score(y_test, prediction_lr, average='micro')
prec = precision_score(y_test, prediction_lr, average='micro')
recall = recall_score(y_test, prediction_lr, average='micro')

#Acc
train_acc_lr = logreg.score(X_train, y_train)*100
test_acc_lr = logreg.score(X_test, y_test)*100

#Random Forest
rf_reg = RandomForestClassifier()
rf_reg.fit(X_train, y_train)
prediction_rf = rf_reg.predict(X_test)
f1_rf = f1_score(y_test, prediction_rf, average='micro')
prec_rf = precision_score(y_test, prediction_rf, average='micro')
recall_rf = recall_score(y_test, prediction_rf, average='micro')

#Acc
train_acc_rf = rf_reg.score(X_train, y_train)*100
test_acc_rf = rf_reg.score(X_test, y_test)*100

#CM & FI
def plot_cm(cm, target_name, title="Confusion Matrix", cmap=None, normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = cmap or plt.get_cmap('Blues')
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_name is not None:
        tick_marks = np.arange(len(target_name))
        plt.xticks(tick_marks, target_name, rotation=45)
        plt.yticks(tick_marks, target_name)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('ConfusionMatrix.png', dpi=120)
    plt.show()

target_name = np.array(['setosa', 'versicolor', 'virginica'])
plot_cm(cm, target_name, title="Confusion Matrix", cmap=None, normalize=True)

#Feature Importance
importances = rf_reg.feature_importances_
labels = dataset.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns=["feature", "importances"])
features = feature_df.sort_values(by='importances', ascending=False)

axis=14
title=12
plt.figure(figsize=(12, 6))
ax= sns.barplot(x='importances', y='feature', data=features)
ax.set_xlabel('Importance', fontsize=axis)
ax.set_ylabel('Feature', fontsize=axis)
ax.set_title('Random Forest Feature Importance', fontsize=title)
plt.tight_layout()
plt.savefig('FeatureImportance.png', dpi=120)
plt.close()

with open('scores.txt') as score:
    score.write(f"Random Forest Train Var: 2.1f%%\n" % train_acc_rf)
    score.write(f"Random Forest Test Var: 2.1f%%\n" % test_acc_rf)
    score.write(f"F1 Score: 2.1f%%\n" % f1_rf)
    score.write(f"Recall Score: 2.1f%%\n" % recall_rf)
    score.write(f"Precision Score: 2.1f%%\n" % prec_rf)

    score.write("\n")
    score.write("\n")

    score.write(f"Logistic Regression Train Var: 2.1f%%\n" % train_acc_lr)
    score.write(f"Logistic Regression Test Var: 2.1f%%\n" % test_acc_lr)
    score.write(f"F1 Score: 2.1f%%\n" % f1)
    score.write(f"Recall Score: 2.1f%%\n" % recall)
    score.write(f"Precision Score: 2.1f%%\n" % prec)
