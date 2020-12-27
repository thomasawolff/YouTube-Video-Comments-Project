


import random
import pickle
import textwrap
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from datetime import datetime
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

url = (r'C:\Users\moose_f8sa3n2\Google Drive\Research Methods\Course Project\YouTube Data\Unicode Files\youTubeVideosUTF.csv')

class textAnalytics(object):

    def __init__(self,file1,
                 numClusters=3,
                 dataFeature1=None,
                 dataFeature2=None,
                 dataFeature3=None,
                 dataFeature4=None,
                 ):
        self.number_clusters = numClusters
        self.dataFeature1 = dataFeature1
        self.dataFeature2 = dataFeature2
        self.dataFeature3 = dataFeature3
        self.dataFeature4 = dataFeature4
        data_df = pd.read_csv(file1,low_memory=False)
        self.token_pattern = '(?u)\\b\\w+\\b'
        review_df_All = data_df[[self.dataFeature1,self.dataFeature2,self.dataFeature3,self.dataFeature4]]
        videoTitles = pd.read_csv('YouTubeVideoTitles.csv')
        self.dataComm = pd.merge(videoTitles, review_df_All, on = dataFeature1)
        self.stopWords = stopwords.words('english')
                

    def sentimentAnalysis(self):
        pol = []
        sub = []
        for i in self.dataComm.commentText.values:
            try:
                analysis = TextBlob(i)
                pol.append(round(analysis.sentiment.polarity,2))
            except:
                pol.append(0)

        for i in self.dataComm.commentText.values:
            try:
                analysis = TextBlob(i)
                sub.append(round(analysis.sentiment.subjectivity,2))
            except:
                sub.append(0)
        self.dataComm['polarity']=pol
        self.dataComm['subjectivity']=sub
        #print(self.dataComm['polarity'])
        self.dataComm.loc[self.dataComm['polarity'] < 0, 'sentimentBucket'] = -1
        self.dataComm.loc[self.dataComm['polarity'] == 0, 'sentimentBucket'] = 0
        self.dataComm.loc[self.dataComm['polarity'] > 0, 'sentimentBucket'] = 1
        #dataComm .to_csv('youTubeVideosSentimentAnalysisSample10000.csv',sep=',',encoding='utf-8')
        ##                    videoID       categoryID  views  ...    replies  polarity   subjectivity
        ##          251449  LLGENw4C1jk          17   1002386  ...      0.0      0.50          0.50
        ##          39834   3VVnY86ulA8          22    802134  ...      0.0      0.00          0.10
        ##          203460  iA86imHKCMw          17   3005399  ...      0.0     -0.08          0.69
        ##          345225  RRkdV_xmYOI          23    367544  ...      0.0      0.13          0.76
        ##          402953  vQ3XgMKAgxc          10  51204658  ...      0.0      0.25          0.50
        

        
    def dataModify(self):
        self.sentimentAnalysis()
        self.dataComm  = self.dataComm[[self.dataFeature1,self.dataFeature2,self.dataFeature3,self.dataFeature4,\
                                        'polarity','subjectivity','sentimentBucket']]
        self.X = self.dataComm.iloc[:,[self.dataComm.columns.get_loc('polarity'),self.dataComm.columns.get_loc('subjectivity')]].values


      
    def kMeansClustering(self):
        self.dataModify()
        kmeans = KMeans(self.number_clusters, init = 'k-means++',max_iter=300,n_init=10)
        self.dataComm['clusters'] = kmeans.fit_predict(self.X)
        return self.dataComm


go = textAnalytics(url,
                   numClusters = 3,
                   dataFeature1 = 'videoID', 
                   dataFeature2 = 'categoryID',
                   dataFeature3 = 'views',
                   dataFeature4 = 'commentText')


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
    sentimentPivot = sentimentPivot.iloc[:,[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,39,40]]
    sentimentPivot.to_csv('SentimentPartition.csv')
    subjectivityPivot = subjectivityPivot.iloc[:,[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,39,40]]
    subjectivityPivot.to_csv('SubjectivityPartition.csv')
    clustersPivot = clustersPivot.iloc[:,[20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,39,40]]
    clustersPivot.to_csv('ClustersPartition.csv')
    return data

#pandasAggregate()


def dataMerge():
    df = pandasAggregate()
    np.seterr(divide = 'ignore')
    #df = pd.read_csv('youTubeVideosUTF.csv',low_memory=False)
    df = df[['videoID','views','categoryID']].drop_duplicates()
    df = df.set_index('videoID')
   
    clusters = pd.read_csv('ClustersPartition.csv')
    clusters['41'].replace('', np.nan, inplace=True)
    clusters.dropna(subset=['41'], inplace=True)
    
    subject = pd.read_csv('SubjectivityPartition.csv')
    subject['41'].replace('', np.nan, inplace=True)
    subject.dropna(subset=['41'], inplace=True)
    
    sentiment = pd.read_csv('SentimentPartition.csv')
    sentiment['41'].replace('', np.nan, inplace=True)
    sentiment.dropna(subset=['41'], inplace=True)

    merge2 = pd.merge(df, clusters, on = 'videoID')
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

    #try: 
    modelPCA = PCA(n_components = 2).fit(X_train)
    print('Variance Explained by PCA model:',modelPCA.explained_variance_ratio_)
    print('Singlular values of PCA model:',modelPCA.singular_values_)
    modelLR = LogisticRegression()
    #except ValueError: pass

    # performing principle components analysis to reduce the number of fields
    # and use the eigenvalues as the data for modeling
    X_train_PCA = modelPCA.transform(X_train)

    # performing Logistic regression on the new PCA model
    modelLR.fit(X_train_PCA,y_train)

    print('Train Performance Logistic Regression with PCA: '+str(round(modelLR.score(X_train_PCA,y_train),2)))
    predictions = modelLR.predict(X_train_PCA)
    print(confusion_matrix(y_train,predictions))

    if operation == 'cross validation':
        print('Cross Validation scores from 8 iterations:')
        scores = cross_val_score(modelLR, X_train_PCA, y_train, cv=8)
        print(scores)
    
    elif operation == 'validation set':
        X_val_PCA = modelPCA.transform(X_val)
        predictions = modelLR.predict(X_val_PCA)
        print('Validation Performance Logistic Regression with PCA: '+str(round(+modelLR.score(X_val_PCA,y_val),2)))
        print('Confusion Matrix:')
        print(confusion_matrix(y_val,predictions))
        print('Classification Report:')
        print(classification_report(y_val,predictions))
        
    elif operation == 'test set':
        X_test_PCA = modelPCA.transform(X_test)
        predictions = modelLR.predict(X_test_PCA)
        print('Test Performance Logistic Regression with PCA: '+str(round(+modelLR.score(X_test_PCA,y_test),2)))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test,predictions))
        print('Classification Report:')
        print(classification_report(y_test,predictions))


    with open('YouTubeModelPickle','wb') as p:
        pickle.dump(modelLR,p)
    
    input_ = input('Hit Enter to leave')

    ##    Variance Explained by PCA model: [0.84019443 0.11170318]
    ##    Singlular values of PCA model: [286.4529428  104.44704467]
    ##    Train Performance Logistic Regression with PCA: 0.73
    ##    [[542 196]
    ##     [235 622]]
    ##    Test Performance Logistic Regression with PCA: 0.71
    ##    Confusion Matrix:
    ##    [[184  65]
    ##     [ 90 193]]
    ##    Classification Report:
    ##                  precision    recall  f1-score   support
    ##
    ##               1       0.67      0.74      0.70       249
    ##               2       0.75      0.68      0.71       283
    ##
    ##        accuracy                           0.71       532
    ##       macro avg       0.71      0.71      0.71       532
    ##    weighted avg       0.71      0.71      0.71       532


modelPredictionsLR('test set')
    


