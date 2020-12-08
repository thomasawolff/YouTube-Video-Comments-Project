import re
import nltk
import math
import nltk.corpus
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from apyori import apriori
from wordcloud import WordCloud
from scipy.stats import pearsonr
from scipy.stats import kurtosis
from nltk import ne_chunk
from textblob import TextBlob
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from nltk.corpus import stopwords
from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer



class textAnalytics(object):

    def __init__(self,file1,sample):
        self.limit = 100
        self.stringsList = []
        self.file1 = file1
        self.review_df = pd.read_csv(self.file1,low_memory=False)
        #self.review_df = self.review_df['commentText']
        self.token_pattern = '(?u)\\b\\w+\\b'
        self.field = 'commentText'
        #print(list(self.review_df))
        self.review_df = self.review_df[['videoID','categoryID','views','likes','dislikes',\
                                         'commentCount','commentText','commentLikes','replies']]
        self.stopWords = stopwords.words('english')
        self.review_df = self.review_df.sample(sample)
        #print(self.stopWords)

    def bowConverter(self):
        bow_converter = CountVectorizer(token_pattern=self.token_pattern)
        x = bow_converter.fit_transform(self.review_df[self.field])
        self.words = bow_converter.get_feature_names()
        #print(len(words)) ## 29221
        
    def biGramConverter(self):
        bigram_converter = CountVectorizer(ngram_range=(2,2), token_pattern=self.token_pattern)
        x2 = bigram_converter.fit_transform(self.review_df[self.field])
        self.bigrams = bigram_converter.get_feature_names()
        #print(len(bigrams)) ## 368937
        #print(bigrams[-10:])
        ##        ['zuzu was', 'zuzus room', 'zweigel wine'
        ##       , 'zwiebel kräuter', 'zy world', 'zzed in'
        ##       , 'éclairs napoleons', 'école lenôtre', 'ém all', 'òc châm']

    def triGramConverter(self):
        trigram_converter = CountVectorizer(ngram_range=(3,3), token_pattern=self.token_pattern)
        x3 = trigram_converter.fit_transform(self.review_df[self.field])
        self.trigrams = trigram_converter.get_feature_names()
        print(len(self.trigrams)) # 881609
        #print(self.trigrams[:10])
        ##        ['0 0 eye', '0 20 less', '0 39 oz', '0 39 pizza', '0 5 i'
        ##         , '0 50 to', '0 6 can', '0 75 oysters', '0 75 that', '0 75 to']

    def gramPlotter(self):
        self.bowConverter()
        self.biGramConverter()
        self.triGramConverter()
        
        sns.set_style("darkgrid")
        counts = [len(self.words), len(self.bigrams), len(self.trigrams)]
        plt.plot(counts, color='cornflowerblue')
        plt.plot(counts, 'bo')
        plt.margins(0.1)
        plt.xticks(range(3), ['unigram', 'bigram', 'trigram'])
        plt.tick_params(labelsize=14)
        plt.title('Number of ngrams in the first 10,000 reviews of the dataset', {'fontsize':16})
        plt.show()


    def wordLem(self):
        self.bowConverter()
        for line in self.words:
            print(line+":"+lemmatizer.lemmatize(line))


    def wordCount(self):
        for line in self.review_df[self.field]:
            wordsTokens = word_tokenize(line)
            self.stringsList.append(Counter(wordsTokens))
        ##print(self.stringsList)
        ##  Counter({'.': 11, 'the': 9, 'and': 8, 'was': 8, 'It': 5, 'I': 5, 'it': 4, 'their': 4


    def stringCleaning(self):
        self.wordCount()
        lengthList = []
        punctuationList = ['-?','!',',',':',';','()',"''",'.',"``",'|','^','..','...','--','=']
        for i in range(0,self.limit):
            try:
                for words in self.stringsList[i]:
                    if len(words)>0:
                        lengthList.append(words)
            except IndexError: pass
        post_punctuation = [word for word in lengthList if word not in punctuationList]
        noStopWords = [word for word in post_punctuation if word not in self.stopWords]
        self.postPunctCount = Counter(noStopWords)
        # print(self.postPunctCount)
        ##        Counter({'I': 9, "n't": 6, 'The': 5, 'go': 5, 'good': 5, "'s": 5,
        ##                 'My': 4, 'It': 4, 'place': 4, 'menu': 4, ')': 4, 'outside': 3,
        ##                 'food': 3, 'like': 3, "'ve": 3, 'amazing': 3, 'delicious': 3,
        ##                 'came': 3, 'wait': 3, 'back': 3, 'They': 3, 'evening': 3, 'try': 3,
        ##                 'one': 3, '(': 3, 'awesome': 3,'much': 3, 'took': 2, 'made': 2,
        ##                 'sitting': 2, 'Our': 2, 'arrived': 2, 'quickly': 2, 'looked': 2, ....


    def tagsMaker(self):
        # If you want to run this code, install Ghostscript first
        self.stringCleaning()
        tags = nltk.pos_tag(self.postPunctCount)
        grams = ne_chunk(tags)
        grammers = r"NP: {<DT>?<JJ>*<NN>}"
        chunk_parser = nltk.RegexpParser(grammers)
        chunk_result = chunk_parser.parse(grams)
        print(chunk_result)
        ##        (ORGANIZATION General/NNP Manager/NNP Scott/NNP Petello/NNP)
        ##          (NP egg/NN)
        ##          Not/RB
        ##          (NP detail/JJ assure/NN)
        ##          albeit/IN
        ##          (NP rare/JJ speak/JJ treat/NN)
        ##          (NP guy/NN)
        ##          (NP respect/NN)
        ##          (NP state/NN)
        ##          'd/MD
        ##          surprised/VBN
        ##          walk/VB
        ##          totally/RB
        ##          satisfied/JJ
        ##          Like/IN
        ##          always/RB
        ##          say/VBP
        ##          (PERSON Mistakes/NNP)
    

    def sentimentAnalysis(self):
        self.stringCleaning()
        pol = []
        sub = []
        self.comm = self.review_df 
        for i in self.comm.commentText.values:
            try:
                analysis = TextBlob(i)
                pol.append(round(analysis.sentiment.polarity,2))
            except:
                pol.append(0)

        for i in self.comm.commentText.values:
            try:
                analysis = TextBlob(i)
                sub.append(round(analysis.sentiment.subjectivity,2))
            except:
                sub.append(0)
        self.comm['polarity']=pol
        self.comm['subjectivity']=sub
        self.comm.loc[self.comm['polarity'] < 0, 'sentimentBucket'] = -1
        self.comm.loc[self.comm['polarity'] == 0, 'sentimentBucket'] = 0
        self.comm.loc[self.comm['polarity'] > 0, 'sentimentBucket'] = 1
        #self.comm.to_csv('youTubeVideosSentimentAnalysisSample10000.csv',sep=',',encoding='utf-8')
        #print(self.comm)
        ##                    videoID       categoryID  views  ...    replies  polarity   subjectivity
        ##          251449  LLGENw4C1jk          17   1002386  ...      0.0      0.50          0.50
        ##          39834   3VVnY86ulA8          22    802134  ...      0.0      0.00          0.10
        ##          203460  iA86imHKCMw          17   3005399  ...      0.0     -0.08          0.69
        ##          345225  RRkdV_xmYOI          23    367544  ...      0.0      0.13          0.76
        ##          402953  vQ3XgMKAgxc          10  51204658  ...      0.0      0.25          0.50
        


    def distPlotter(self):
        self.sentimentAnalysis()
        name1 = str('commentCount')
        field = self.comm[name]
        print(round(field.drop_duplicates().describe(include='all')),2)
        print('Kurtosis:',round(kurtosis(field),2))
        print('Pearson R Correlation Views/Comments:',pearsonr(self.comm['polarity'],self.comm['subjectivity']))
        #print('Pearson R Correlation Views/Likes:',pearsonr(self.comm['views'],self.comm['likes']))
        #print('Pearson R Correlation Views/Dislikes:',pearsonr(self.comm['views'],self.comm['dislikes']))
        plt.grid(axis='y', alpha=0.50)
        plt.title('Histogram of '+name1)
        plt.xlabel(name2)
        plt.ylabel('Subjectivity')
        plt.hist(field,bins=70)
        plt.ticklabel_format(style='plain')
        #sns.distplot(self.comm['views'],hist=True,fit=norm,kde=False,norm_hist=False)
        #x,y = sns.kdeplot(self.comm['views']).get_lines()[0].get_data()
        plt.show()

        
    def dataModify(self):
        self.number_clusters = 5
        self.sentimentAnalysis()
        self.comm['views'] = np.log2(self.comm['views'])
        self.comm = self.comm[['videoID','categoryID','views','commentText','polarity','subjectivity','sentimentBucket']].copy()
        column1 = 4
        column2 = 5
        self.X = self.comm.iloc[:,[column1,column2]].values
        #print(self.X)


    def dataReturn(self):
        commOut = self.review_df[['videoID','views','categoryID']].copy()
        return commOut


    def dendrogram(self,linkage):
        self.dataModify()
        # using dendrogram to optimal number of clusters
        dendrogram = sch.dendrogram(sch.linkage(self.X,linkage))
        plt.title('Dendrogram')
        plt.xlabel('Sentiment Value')
        plt.ylabel('Subjectivity Value')
        plt.axis('off')
        plt.show()

      
    def kMeansElbow(self):
        self.dataModify()
        # using the elbow method to find optimal number of clusters
        wcss = []
        for i in range(1, 11):
           kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300,n_init=10,random_state=0)
           kmeans.fit(self.X)
           wcss.append(kmeans.inertia_)   
        plt.plot(range(1, 11),wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of Data Clusters')
        plt.ylabel('WCSS')
        plt.show()

      
    def kMeansClustering(self):
        self.dataModify()
        # applying k means to the dataset
        self.kmeans = KMeans(n_clusters = self.number_clusters, init = 'k-means++',max_iter=300,n_init=10)
        self.y_kmeans = self.kmeans.fit_predict(self.X)
        self.comm['clusters'] = self.y_kmeans
        self.clust1 = self.comm.loc[self.comm['clusters'] == 0]
        self.clust2 = self.comm.loc[self.comm['clusters'] == 1]
        self.clust3 = self.comm.loc[self.comm['clusters'] == 2]
        self.clust4 = self.comm.loc[self.comm['clusters'] == 3]
        self.clust5 = self.comm.loc[self.comm['clusters'] == 4]
        self.commNums = self.comm[['videoID','categoryID','views','clusters','polarity','subjectivity','sentimentBucket']].copy()
        return self.commNums
        #self.commNums.to_csv('youTubeVideosSentimentAnalysisOutput.csv',sep=',',encoding='utf-8')


    def dataReturnClusters(self):
        self.kMeansClustering()
        commOut = self.review_df[['videoID','views','categoryID','commentText']].copy()
        commOut['clusters'] = self.comm['clusters']
        return commOut
       

    def kMeansVisualizer(self):
        self.kMeansClustering()
        # visualizing the clusters using K-means
        for i in range(0,self.number_clusters):
           plt.scatter(self.X[self.y_kmeans == i, 0], self.X[self.y_kmeans == i, 1], s = 100)
           #plt.scatter(self.kmeans.cluster_centers_[:,0],self.kmeans.cluster_centers_[:,1],s=50,c='yellow',label='Centroids')
        plt.title('Clusters of Sentiment vs Subjectivity: K-Means Method')
        plt.xlabel('Sentiment Value')
        plt.ylabel('Subjectivity Value')
        plt.legend(set(self.y_kmeans))
        plt.show()


    def wordCloudVisualizer(self):
        self.kMeansClustering()
        wordcloud = WordCloud(
            background_color='white',
            stopwords= ["dtype","commentText"]+stopwords.words('english'),
            max_words=200,
            max_font_size=40, 
            scale=3
            ).generate(str(self.clust1['commentText']))
        fig = plt.figure(1, figsize=(6,6))
        plt.title('Word cloud of chosen cluster')
        plt.axis('off')
        plt.imshow(wordcloud)
        plt.show()

        
      


url = (r'C:\Users\moose_f8sa3n2\Google Drive\Research Methods\Course Project\YouTube Data\Unicode Files\youTubeVideosUTF.csv')
affinity = 'euclidean'
linkage = 'ward'





