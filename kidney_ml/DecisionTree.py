import warnings
import graphviz
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn import model_selection
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import datasets, linear_model, tree, ensemble
from sklearn.metrics import precision_recall_fscore_support as pr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,\
    recall_score,f1_score, classification_report

warnings.simplefilter(action='ignore', category=FutureWarning)

# Input Dataset
KidneyDataset = pd.read_excel("FinalKidneyDatasetWithoutFarValues.xlsx")
KidneyDataset = KidneyDataset.fillna(method='ffill')

X = KidneyDataset[['Mean','Minimum','Maximum','VoxelNum','VolumeNum','VoxelNum2','VolumeNum2',
                 'Mean2','Minimum2','Maximum2','VoxelVolume','Maximum3DDiameter','MeshVolume',
                 'MajorAxisLength','Sphericity','LeastAxisLength','Elongation','SurfaceVolumeRatio',
                 'Maximum2DDiameterSlice','Flatness','SurfaceArea','MinorAxisLength',
                 'Maximum2DDiameterColumn','Maximum2DDiameterRow','GrayLevelVariance',
                 'HighGrayLevelEmphasis', 'DependenceEntropy','DependenceNonUniformity',
                 'GrayLevelNonUniformity','SmallDependenceEmphasis', 'SmallDependenceHighGrayLevelEmphasis',
                 'DependenceNonUniformityNormalized','LargeDependenceEmphasis',
                 'LargeDependenceLowGrayLevelEmphasis','DependenceVariance',
                 'LargeDependenceHighGrayLevelEmphasis', 'SmallDependenceLowGrayLevelEmphasis',
                 'LowGrayLevelEmphasis','JointAverage','SumAverage','JointEntropy', 'ClusterShade',
                 'MaximumProbability','Idmn','JointEnergy','Contrast','DifferenceEntropy','InverseVariance',
                 'DifferenceVariance','Idn','Idm','Correlation','Autocorrelation','SumEntropy',
                 'MCC','SumSquares','ClusterProminence', 'Imc2','Imc1','DifferenceAverage','Id',
                 'ClusterTendency','InterquartileRange','Skewness','Uniformity','Median','Energy',
                 'RobustMeanAbsoluteDeviation','MeanAbsoluteDeviation','TotalEnergy','Maximum3',
                 'RootMeanSquared','90Percentile','Minimum3', 'Entropy','Range','Variance','10Percentile',
                 'Kurtosis','Mean3','ShortRunLowGrayLevelEmphasis','GrayLevelVariance2',
                 'LowGrayLevelRunEmphasis', 'GrayLevelNonUniformityNormalized','RunVariance',
                 'GrayLevelNonUniformity2','LongRunEmphasis','ShortRunHighGrayLevelEmphasis',
                 'RunLengthNonUniformity', 'ShortRunEmphasis','LongRunHighGrayLevelEmphasis',
                 'RunPercentage','LongRunLowGrayLevelEmphasis','RunEntropy','HighGrayLevelRunEmphasis',
                 'RunLengthNonUniformityNormalized','GrayLevelVariance3', 'ZoneVariance',
                 'GrayLevelNonUniformityNormalized2','SizeZoneNonUniformityNormalized',
                 'SizeZoneNonUniformity','GrayLevelNonUniformity3','LargeAreaEmphasis',
                 'SmallAreaHighGrayLevelEmphasis', 'ZonePercentage','LargeAreaLowGrayLevelEmphasis',
                 'LargeAreaHighGrayLevelEmphasis','HighGrayLevelZoneEmphasis','SmallAreaEmphasis',
                 'LowGrayLevelZoneEmphasis','ZoneEntropy', 'SmallAreaLowGrayLevelEmphasis',
                 'Coarseness','Complexity','Strength','Contrast2','Busyness','Sex','Age','WBC',
                 'RBC','PLT','Cr','Urea','eGFR', 'UrineVol24h','UrinaryPor24h','UrinaryCr24h',
                 'UrinrProICr','RTfield','Kidney','KidneyDose','StandardDeviation','Volum',
                 'TotalDose(Gy)','Beam','Fraction','geGFR','Dose','volume']]

y = KidneyDataset['DamageLabel']

# Model
X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=42)
clf = DecisionTreeClassifier(max_depth=10, max_features=4, min_samples_leaf=4,
                             presort=True, random_state=4, max_leaf_nodes=5).fit(X, y)
clf.fit(X, y)
print('accuracy of RF for KidneyDataset on training set is: {:.2f}'.format(clf.score(X, y)))
print('accuracy of RF for KidneyDataset on testing set is: {:.2f}'.format(clf.score(X_test, y_test)))
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
total = sum(sum(cm))

# Calculate evaluation criteria
accuracy = (cm[0, 0] + cm[1, 1]) / total
print('accuracy:', accuracy)
sensivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('sensivity:', sensivity)
specifity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('specifity:', specifity)

cv_scores = cross_val_score(clf, X, y)
print('cross validation scores:', cv_scores)
print('mean cross validation score: {:.3f}'.format(np.mean(cv_scores)))

print(classification_report(y_test, y_pred, target_names=['not 1', '1']))
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = DecisionTreeClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print('accuracy:')
print(results.mean(), results.std())
accuracy=(cross_val_score(clf,X,y,scoring='accuracy',cv=5).mean()*100)
print(accuracy)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, clf.predict_proba(X)[:, 1])
print('auc:')
print(auc(false_positive_rate, true_positive_rate))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, clf.predict_proba(X)[:, 1])

# Find and sort important features
feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']
                                   ).sort_values('importance',ascending=False)
print(feature_importances.to_string())
with open('DT_feature_importances.txt', 'a') as the_file:
    the_file.write(str(feature_importances.to_string()))
