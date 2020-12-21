
from textAnalyticsAppRun import *

import pyodbc
import random
import pickle
import textwrap
from datetime import datetime
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



def pandasAggregate():
    data = go.kMeansClustering()
    
    dataPolarity = data[['videoID','sentimentBucket']].copy()
    dataSubjectivity = data[['videoID','subjectivity']].copy()
    dataClusters = data[['videoID','clusters']].copy()

    # this code partitions the data by video ID and counts the number of values in the sentiment bucket column
    # giving each row an incremented value which is then used for the pivot of the data
    dataPolarity['dataRowNumSentiment'] = dataPolarity.sort_values(['videoID','sentimentBucket'], ascending=[True,False])\
             .groupby(['videoID'])\
             .cumcount() + 1

    # this code partitions the data by video ID and counts the number of values in the subjectivity column
    # giving each row an incremented value which is then used for the pivot of the data
    dataSubjectivity['dataRowNumSubjectivity'] = dataSubjectivity.sort_values(['videoID','subjectivity'], ascending=[True,False])\
             .groupby(['videoID'])\
             .cumcount() + 1

    # this code partitions the data by video ID and counts the number of values in the clusters column
    # giving each row an incremented value which is then used for the pivot of the data
    dataClusters['dataRowNumClusters'] = dataClusters.sort_values(['videoID','clusters'], ascending=[True,False])\
             .groupby(['videoID'])\
             .cumcount() + 1

    # this code pivots the data using the fields created above. All values in these fields with be on one row per video ID
    sentimentPivot = dataPolarity.pivot(index='videoID', columns='dataRowNumSentiment', values='sentimentBucket')
    subjectivityPivot = dataSubjectivity.pivot(index='videoID', columns='dataRowNumSubjectivity', values='subjectivity')
    clustersPivot = dataClusters.pivot(index='videoID', columns='dataRowNumClusters', values='clusters')


    # I split up this part of the project into csv files since running all this data through the
    # pipeline would take a long time especially when I was developing the predictive model.
    sentimentPivot.to_csv('SentimentPartition.csv')
    subjectivityPivot.to_csv('SubjectivityPartition.csv')
    clustersPivot.to_csv('ClustersPartition.csv')

#pandasAggregate()


def dataMerge():
    np.seterr(divide = 'ignore')
    go = textAnalytics(url)
    df = go.dataReturnAll()[['videoID','views','categoryID']].drop_duplicates()
    df = df.set_index('videoID')
    
    encoded = pd.read_csv('sentencesEncoded2.csv')
    clusters = pd.read_csv('ClustersPartitionFinal20.csv')
    subject = pd.read_csv('SubjectivityPartitionFinal20.csv')
    sentiment = pd.read_csv('SentimentPartitionFinal20.csv')

    merge1 = pd.merge(df, encoded, on = 'videoID')
    merge2 = pd.merge(merge1, clusters, on = 'videoID')
    merge3 = pd.merge(merge2, subject, on = 'videoID')
    merge4 = pd.merge(merge3, sentiment, on = 'videoID')

    # doing log transform of the views field
    merge4['views'] = np.log2(merge4['views'])

    # creating value buckets for the views field which will become a target variable for the model
    merge4.loc[merge4['views'] < 20, 'viewsBucket'] = '1'
    #merge4.loc[(merge4['views'] > 18) & (merge4['views'] <= 20), 'viewsBucket'] = '2'
    #merge4.loc[(merge4['views'] > 20) & (merge4['views'] <= 22), 'viewsBucket'] = '3'
    merge4.loc[merge4['views'] > 20, 'viewsBucket'] = '2'

    #print(round(merge4['views'].describe(include='all')),2)
    ##    25%        18.0
    ##    50%        20.0
    ##    75%        22.0
    ##    max        29.0
    ##    Name: views, dtype: float64 2
    
    merge4 = merge4.set_index('videoID')
    del merge4['views']
    merge4.to_csv('dataCombined.csv')
    
    return merge4

    ##    videoID        categoryID views  ...   SentimentKBucket40  viewsBucket                                
    ##    _0d3XbH12cs          10   18.0  ...                   1       2
    ##    _38JDGnr0vA          15   24.0  ...                   1       4
    ##    _4PLKxYZUPc          22   21.0  ...                   1       3
    ##    _5wCA9OM00o          22   19.0  ...                   1       2
    ##    _5ZrSKpbdSg          28   19.0  ...                   1       2
    ##    ...                 ...    ...  ...                 ...          ...
    ##    zyPIdeF4NFI          22   19.0  ...                   1       2
    ##    ZYQ1cVRtMZU          26   24.0  ...                   1       4
    ##    ZYSjPZUqLdk          22   21.0  ...                   1       3
    ##    zZ2CLmvqfXg          24   22.0  ...                   1       3
    ##    Z-zdIGxOJ4M          10   19.0  ...                   1       2
    ##
    ##    [2661 rows x 69 columns]

#print(dataMerge())

def modelPredictionsLR(operation):
    data = dataMerge()
    
    X_train, X_test, y_train, y_test = train_test_split(data, data['viewsBucket'], test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    # deleting the viewsBucket field from the X train, validate, and test sets
    del X_train['viewsBucket']
    del X_test['viewsBucket']
    del X_val['viewsBucket']

    try: 
        modelPCA = PCA(n_components = 2).fit(X_train)
        print('Variance Explained by PCA model:',modelPCA.explained_variance_ratio_)
        print('Singlular values of PCA model:',modelPCA.singular_values_)
        modelLR = LogisticRegression()
    except ValueError: pass

    # performing principle components analysis to reduce the number of fields
    # and use the eigenvalues as the data for modeling
    X_train_PCA = modelPCA.transform(X_train)

    # performing Logistic regression on the new PCA model
    modelLR.fit(X_train_PCA,y_train)

    print('Train Performance Logistic Regression with PCA: '+str(round(modelLR.score(X_train_PCA,y_train),2)))
    predictions = modelLR.predict(X_train_PCA)
    print(confusion_matrix(y_train,predictions))
    
    if operation == 'validation':
        X_val_PCA = modelPCA.transform(X_val)
        predictions = modelLR.predict(X_val_PCA)
        print('Validation Performance Logistic Regression with PCA: '+str(round(+modelLR.score(X_val_PCA,y_val),2)))
        print('Confusion Matrix:')
        print(confusion_matrix(y_val,predictions))
        print('Classification Report:')
        print(classification_report(y_val,predictions))
        
    elif operation == 'test':
        X_test_PCA = modelPCA.transform(X_test)
        predictions = modelLR.predict(X_test_PCA)
        print('Test Performance Logistic Regression with PCA: '+str(round(+modelLR.score(X_test_PCA,y_test),2)))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test,predictions))
        print('Classification Report:')
        print(classification_report(y_test,predictions))

    #print('Cross Validation scores from 8 iterations:')
    #scores = cross_val_score(modelLR, X_train_PCA, y_train, cv=8)
    #print(scores)

    with open('YouTubeModelPickle','wb') as p:
        pickle.dump(modelLR,p)
    
    input_ = input('Hit Enter to leave')

    ##    Variance Explained by PCA model: [0.8362525 0.1170726]
    ##    Singlular values of PCA model: [285.0068869  106.63855784]
    ##    Train Performance Logistic Regression with PCA: 0.71
    ##    [[415 332]
    ##     [137 712]]
    ##    Validation Performance Logistic Regression with PCA: 0.71
    ##    Confusion Matrix:
    ##    [[148 100]
    ##     [ 52 232]]
    ##    Classification Report:
    ##                  precision    recall  f1-score   support
    ##
    ##               1       0.74      0.60      0.66       248
    ##               2       0.70      0.82      0.75       284
    ##
    ##        accuracy                           0.71       532
    ##       macro avg       0.72      0.71      0.71       532
    ##    weighted avg       0.72      0.71      0.71       532
    ##
    ##    Cross Validation scores from 8 iterations:
    ##    [0.685   0.74   0.76   0.725   0.69   0.69   0.668   0.69]



    


