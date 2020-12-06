import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,\
    recall_score,f1_score, classification_report

warnings.simplefilter(action='ignore', category=FutureWarning)

# Input Dataset
BookReviews = pd.read_excel("E:\Academic\Kidney\DataSet\FinalKidneyDatasetWithoutFarValues.xlsx")
BookReviews = BookReviews.fillna(method='ffill')

X = BookReviews[['Mean','Minimum','Maximum','VoxelNum','VolumeNum','VoxelNum','VolumeNum',
                 'Mean2','Minimum2','Maximum2','VoxelVolume','Maximum3DDiameter','MeshVolume',
                 'MajorAxisLength','Sphericity','LeastAxisLength','Elongation','SurfaceVolumeRatio',
                 'Maximum2DDiameterSlice','Flatness','SurfaceArea','MinorAxisLength',
                 'Maximum2DDiameterColumn','Maximum2DDiameterRow','GrayLevelVariance',
                 'HighGrayLevelEmphasis', 'DependenceEntropy','DependenceNonUniformity',
                 'GrayLevelNonUniformity','SmallDependenceEmphasis', 'SmallDependenceHighGrayLevelEmphasis',
                 'DependenceNonUniformityNormalized','LargeDependenceEmphasis',
                 'LargeDependenceLowGrayLevelEmphasis','DependenceVariance','LargeDependenceHighGrayLevelEmphasis',
                 'SmallDependenceLowGrayLevelEmphasis','LowGrayLevelEmphasis','JointAverage','SumAverage',
                 'JointEntropy', 'ClusterShade','MaximumProbability','Idmn','JointEnergy','Contrast',
                 'DifferenceEntropy','InverseVariance','DifferenceVariance','Idn','Idm','Correlation',
                 'Autocorrelation','SumEntropy','MCC','SumSquares','ClusterProminence','Imc2','Imc1',
                 'DifferenceAverage','Id','ClusterTendency','InterquartileRange','Skewness','Uniformity',
                 'Median','Energy','RobustMeanAbsoluteDeviation','MeanAbsoluteDeviation','TotalEnergy',
                 'Maximum3','RootMeanSquared','90Percentile','Minimum3','Entropy','Range','Variance',
                 '10Percentile','Kurtosis','Mean3','ShortRunLowGrayLevelEmphasis','GrayLevelVariance',
                 'LowGrayLevelRunEmphasis','GrayLevelNonUniformityNormalized','RunVariance',
                 'GrayLevelNonUniformity','LongRunEmphasis','ShortRunHighGrayLevelEmphasis',
                 'RunLengthNonUniformity','ShortRunEmphasis','LongRunHighGrayLevelEmphasis',
                 'RunPercentage','LongRunLowGrayLevelEmphasis','RunEntropy','HighGrayLevelRunEmphasis',
                 'RunLengthNonUniformityNormalized','GrayLevelVariance','ZoneVariance',
                 'GrayLevelNonUniformityNormalized','SizeZoneNonUniformityNormalized','SizeZoneNonUniformity',
                 'GrayLevelNonUniformity','LargeAreaEmphasis','SmallAreaHighGrayLevelEmphasis',
                 'ZonePercentage','LargeAreaLowGrayLevelEmphasis','LargeAreaHighGrayLevelEmphasis',
                 'HighGrayLevelZoneEmphasis','SmallAreaEmphasis','LowGrayLevelZoneEmphasis','ZoneEntropy',
                 'SmallAreaLowGrayLevelEmphasis','Coarseness','Complexity','Strength','Contrast','Busyness',
                 'Sex','Age','WBC','RBC','PLT','Cr','Urea','eGFR','UrineVol24h','UrinaryPor24h','UrinaryCr24h',
                 'UrinrProICr','RTfield','Kidney','KidneyDose','StandardDeviation','Volum','TotalDose(Gy)','Beam',
                 'Fraction','geGFR','Dose','volume','Sex','Age','WBC','RBC','PLT','Cr','Urea','eGFR','UrineVol24h',
                 'UrinaryPor24h','UrinaryCr24h','UrinrProICr','RTfield','Kidney','KidneyDose','StandardDeviation',
                 'Volum','TotalDose(Gy)','Beam','Fraction','geGFR']]

y = BookReviews['DamageLabel']

# Model
X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=0)
grbt = GradientBoostingClassifier(n_estimators=3,learning_rate=0.02, max_depth=7, max_features=9)
grbt.fit(X_train,y_train)

# Calculate evaluation criteria
print('accuracy of Gradient Boosting Trees for'
      ' KidneyDataset on training set is: {:.2f}' .format(grbt.score(X_train,y_train)))
print('accuracy of Gradient Boosting Trees for'
      ' KidneyDataset on testing set is: {:.2f}' .format(grbt.score(X_test,y_test)))
y_pred=grbt.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
cv_scores= cross_val_score(grbt,X,y)
print('cross validation scores:',cv_scores)
print('mean cross validation score: {:.3f}'.format(np.mean(cv_scores)))

y_pred = grbt.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
total = sum(sum(cm))
accuracy= (cm[0,0]+cm[1,1])/total
print('accuracy:',accuracy)
sensivity= cm[0,0]/(cm[0,0]+cm[0,1])
print('sensivity:',sensivity)
specifity=cm[1,1]/(cm[1,0]+cm[1,1])
print('specifity:',specifity)

cv_scores= cross_val_score(grbt,X,y)
print('cross validation scores:',cv_scores)
print('mean cross validation score: {:.3f}'.format(np.mean(cv_scores)))

print(classification_report(y_test,y_pred, target_names=['not 1', '1']))
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = DecisionTreeClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print('accuracy:')
print(results.mean(), results.std())

accuracy=(cross_val_score(grbt,X,y,scoring='accuracy',cv=5).mean()*100)
print(accuracy)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, grbt.predict_proba(X)[:,1])
print('auc:')
print(auc(false_positive_rate, true_positive_rate))

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)


print(classification_report(y_test,y_pred, target_names=['not 1', '1']))
accuracy=(cross_val_score(grbt,X,y,scoring='accuracy',cv=10).mean()*100)
features='SummaryPolarity','SummarySubjectivity','%SummaryNumOfNouns','%SummaryNumOfADJ',\
         'SummaryLength','SummaryRichness','NameOfCountry','NumOfHElpfulVotes','NumOFREviews',\
         'ReviewerCountry(Boolean)','AmazonBadges','AboutSection(Boolean)','ReviewerProfilePicture(Boolean)',\
         'ReviewerRanking','ReviewerEmailInReviewerName','MoreInfoinName','overall','ReviewLength',\
         'Readability','%NumOfADJ','%NumberOfNouns','Polarity','Subjectivity','Lexical Diversity(richness)' ,'unixReviewTime'
plt.xticks(range(X.shape[1]),features,  rotation=90)
plt.show()

feature_importances = pd.DataFrame(grbt.feature_importances_, index = X_train.columns,
                                   columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)
