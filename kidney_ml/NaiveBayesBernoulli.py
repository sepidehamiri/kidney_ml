import warnings
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import datasets, linear_model, tree, ensemble
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,\
    recall_score,f1_score, classification_report

warnings.simplefilter(action='ignore', category=FutureWarning)

# Input Dataset
KidneyDataset = pd.read_excel("E:\Academic\Kidney\DataSet\FinalKidneyDataset.xlsx")
KidneyDataset = KidneyDataset.fillna(method='ffill')

X = KidneyDataset[['Mean','Minimum','Maximum','VoxelNum','VolumeNum','VoxelNum','VolumeNum',
                 'Mean2','Minimum2','Maximum2','VoxelVolume','Maximum3DDiameter','MeshVolume',
                 'MajorAxisLength','Sphericity','LeastAxisLength','Elongation','SurfaceVolumeRatio',
                 'Maximum2DDiameterSlice','Flatness','SurfaceArea','MinorAxisLength',
                 'Maximum2DDiameterColumn','Maximum2DDiameterRow','GrayLevelVariance','HighGrayLevelEmphasis',
                 'DependenceEntropy','DependenceNonUniformity','GrayLevelNonUniformity','SmallDependenceEmphasis',
                 'SmallDependenceHighGrayLevelEmphasis','DependenceNonUniformityNormalized','LargeDependenceEmphasis',
                 'LargeDependenceLowGrayLevelEmphasis','DependenceVariance','LargeDependenceHighGrayLevelEmphasis',
                 'SmallDependenceLowGrayLevelEmphasis','LowGrayLevelEmphasis','JointAverage','SumAverage',
                 'JointEntropy', 'ClusterShade','MaximumProbability','Idmn','JointEnergy','Contrast',
                 'DifferenceEntropy','InverseVariance', 'DifferenceVariance','Idn','Idm','Correlation',
                 'Autocorrelation','SumEntropy','MCC','SumSquares','ClusterProminence', 'Imc2','Imc1',
                 'DifferenceAverage','Id','ClusterTendency','InterquartileRange','Skewness','Uniformity','Median',
                 'Energy', 'RobustMeanAbsoluteDeviation','MeanAbsoluteDeviation','TotalEnergy','Maximum3',
                 'RootMeanSquared','90Percentile','Minimum3', 'Entropy','Range','Variance','10Percentile','Kurtosis',
                 'Mean3','ShortRunLowGrayLevelEmphasis','GrayLevelVariance','LowGrayLevelRunEmphasis',
                 'GrayLevelNonUniformityNormalized','RunVariance','GrayLevelNonUniformity','LongRunEmphasis',
                 'ShortRunHighGrayLevelEmphasis','RunLengthNonUniformity', 'ShortRunEmphasis',
                 'LongRunHighGrayLevelEmphasis','RunPercentage','LongRunLowGrayLevelEmphasis','RunEntropy',
                 'HighGrayLevelRunEmphasis','RunLengthNonUniformityNormalized','GrayLevelVariance',
                 'ZoneVariance','GrayLevelNonUniformityNormalized','SizeZoneNonUniformityNormalized',
                 'SizeZoneNonUniformity','GrayLevelNonUniformity','LargeAreaEmphasis','SmallAreaHighGrayLevelEmphasis',
                 'ZonePercentage','LargeAreaLowGrayLevelEmphasis','LargeAreaHighGrayLevelEmphasis',
                 'HighGrayLevelZoneEmphasis','SmallAreaEmphasis','LowGrayLevelZoneEmphasis','ZoneEntropy',
                 'SmallAreaLowGrayLevelEmphasis','Coarseness','Complexity','Strength','Contrast','Busyness','Sex',
                 'Age','WBC','RBC','PLT','Cr','Urea','eGFR', 'UrineVol24h','UrinaryPor24h','UrinaryCr24h',
                 'UrinrProICr','RTfield','Kidney','KidneyDose','StandardDeviation','Volum','TotalDose(Gy)','Beam',
                 'Fraction','geGFR','Dose', 'volume','Sex','Age','WBC','RBC','PLT','Cr','Urea','eGFR','UrineVol24h',
                 'UrinaryPor24h'	,'UrinaryCr24h','UrinrProICr','RTfield','Kidney','KidneyDose','StandardDeviation',
                 'Volum','TotalDose(Gy)','Beam','Fraction','geGFR']]

y = KidneyDataset['DamageLabel']

# Model
X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=1)
nb = BernoulliNB(alpha=1, binarize=1, fit_prior=True, class_prior=None).fit(X_train,y_train)

print('accuracy of NB for KidneyDataset on training set is: {:.2f}' .format(nb.score(X_train,y_train)))
print('accuracy of NB for KidneyDataset on testing set is: {:.2f}' .format(nb.score(X_test,y_test)))
y_pred = nb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

# Calculate evaluation criteria
cv_scores= cross_val_score(nb,X,y)
print('cross validation scores:',cv_scores)
print('mean cross validation score: {:.3f}'.format(np.mean(cv_scores)))
print(classification_report(y_test,y_pred, target_names=['not 1', '1']))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, nb.predict_proba(X)[:, 1])
print('auc:')
print(auc(false_positive_rate, true_positive_rate))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, nb.predict_proba(X)[:, 1])


