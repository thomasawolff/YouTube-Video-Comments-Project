
from textAnalyticsApp import *

import csv
import pyodbc
import random
import pickle
import textwrap
from datetime import datetime
import seaborn as sn
from sklearn.svm import SVC
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


class commentVectors(textAnalytics):

    # calling the dataReturnClusters() method from the textAnalytics class
    def __init__(self,cluster,sampleNum,commentNum):
        self.commentNum = commentNum
        go = textAnalytics(url,sampleNum)
        self.dataComm = go.dataReturnClusters()
        self.cluster = cluster


    def dataReturned(self):
        self.dataComm = self.dataComm.loc[self.dataComm['clusters'] == self.cluster]
        # dont do more than 10 comments in a sample, very computationally intensive
        comments = self.dataComm['commentText'].sample(self.commentNum) 
        self.random = '"'+comments+'"'


    # this project uses GPU processing in tensorflow for embedding the comments
    def embed_useT(self,module):
        with tf.Graph().as_default():
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                    gpus[1],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000),
                      tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    print(e)
        
            sentences = tf.placeholder(tf.string)
            embed = hub.Module(module)
            embeddings = embed(sentences)
            session = tf.train.MonitoredSession()
        return lambda x: session.run(embeddings, {sentences: x})


    # visualizes the heatmap for correlation from inner product between the sentence vectors.
    def plot_similarity(self):
        self.dataReturned()
        embed_fn = self.embed_useT(r'C:\Users\moose_f8sa3n2\Google Drive\Research Methods\Course Project\YouTube Data\Unicode Files')
        encoding_matrix = embed_fn(self.random)
        products = np.inner(encoding_matrix, encoding_matrix)
        mask = np.zeros_like(products, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.set(font_scale=.8)
        g = sns.heatmap(
            products,
            xticklabels=[textwrap.fill(e,15) for e in self.random],
            yticklabels=[textwrap.fill(e,40) for e in self.random],
            vmin=0,
            vmax=1,
            cmap="YlOrRd",
            mask=mask)
        g.figure.set_size_inches(10,8)
        plt.xticks(rotation=0)
        g.set_title("Semantic Textual Similarity")
        plt.show()



# I never used this but once in this project when I was working with data in SQL Server early in the project
def databaseConnection():
    dataList = []
    cnxn = pyodbc.connect(r'Driver={SQL Server};Server=DESKTOP-K3QDR1M\MSSQLSERVER20191;Database=YouTube Data Project;Trusted_Connection=yes;')
    cursor = cnxn.cursor()
    cursor.execute("SELECT * from YouTubeDataPivoted")
    cols = [column[0] for column in cursor.description]
    while 1:
        row = cursor.fetchone()
        if not row:
            break
        dataList.append(list(row))
    cnxn.close()
    df = pd.DataFrame(dataList, columns = cols)
    #print(df)

#databaseConnection()


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
    go = textAnalytics(url,472077)
    df = go.dataReturn()
    df = pd.DataFrame(df)
    df = df[['videoID','views','categoryID']].drop_duplicates()
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

    modelPCA = PCA(n_components = 2).fit(X_train)
    print('Variance Explained by PCA model:',modelPCA.explained_variance_ratio_)
    print('Singlular values of PCA model:',modelPCA.singular_values_)
    modelLR = LogisticRegression()

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

    print('Cross Validation scores from 8 iterations:')
    scores = cross_val_score(modelLR, X_train_PCA, y_train, cv=8)
    print(scores)
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


#modelPredictionsLR()


    


