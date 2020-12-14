import re
import nltk
import math
import json
import nltk.corpus
import operator
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from apyori import apriori
from wordcloud import WordCloud
from scipy.stats import pearsonr
from scipy.stats import kurtosis
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
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


def jSonYield():
    ids = []
    titleList = []
    data = json.load(open('US_category_id.json'))
    pos0 = data.keys()
    pos1 = data.get('items')
    try:
        for i in range(0,len(pos1)):
            id_ = pos1[i]['id']
            title = pos1[i]['snippet']['title']
            ids.append(id_)
            titleList.append(title)
            cats = dict(zip(ids,titleList))
    except IndexError: pass
    return cats



class textAnalytics(object):

    def __init__(self,file1,
                 numClusters=1,
                 KmeansColumn1=None,
                 KmeansColumn2=None,
                 cluster=1,
                 category=None,
                 dataFeature1=None,
                 dataFeature2=None,
                 dataFeature3=None,
                 dataFeature4=None,
                 sentiment = None
                 ):

        dict_ = jSonYield()
        self.limit = 100
        self.stringsList = []
        self.cluster = cluster
        self.sentiment = sentiment
        self.number_clusters = numClusters
        self.KmeansColumn1 = KmeansColumn1
        self.KmeansColumn2 = KmeansColumn2
        self.dataFeature1 = dataFeature1
        self.dataFeature2 = dataFeature2
        self.dataFeature3 = dataFeature3
        self.dataFeature4 = dataFeature4
        categoryPick = pd.DataFrame(dict_.items(),columns=[self.dataFeature2,'category'])
        data_df = pd.read_csv(file1,low_memory=False)
        self.token_pattern = '(?u)\\b\\w+\\b'
        categoryPick[self.dataFeature2] = categoryPick[self.dataFeature2].astype(int)
        review_df_All = data_df[[self.dataFeature1,self.dataFeature2,self.dataFeature3,self.dataFeature4]]
        review_df_All = pd.merge(categoryPick, review_df_All, on = self.dataFeature2)
        review_df_All = review_df_All.loc[review_df_All['category'] == category]
        self.stopWords = stopwords.words('english')
        try:
            print('There are ',len(review_df_All),' comments on this topic')
            self.review_df = review_df_All.sample(10000).drop_duplicates()
        except ValueError:
            print('There are ',len(review_df_All),' comments on this topic')
            self.review_df = review_df_All.drop_duplicates()
            

    def bowConverter(self):
        bow_converter = CountVectorizer(token_pattern=self.token_pattern)
        x = bow_converter.fit_transform(self.review_df[self.dataFeature4])
        self.words = bow_converter.get_feature_names()
        #print(len(words)) ## 29221

        
    def biGramConverter(self):
        bigram_converter = CountVectorizer(ngram_range=(2,2), token_pattern=self.token_pattern)
        x2 = bigram_converter.fit_transform(self.review_df[self.dataFeature4])
        self.bigrams = bigram_converter.get_feature_names()
        #print(len(bigrams)) ## 368937
        #print(bigrams[-10:])
        ##        ['zuzu was', 'zuzus room', 'zweigel wine'
        ##       , 'zwiebel kräuter', 'zy world', 'zzed in'
        ##       , 'éclairs napoleons', 'école lenôtre', 'ém all', 'òc châm']


    def triGramConverter(self):
        trigram_converter = CountVectorizer(ngram_range=(3,3), token_pattern=self.token_pattern)
        x3 = trigram_converter.fit_transform(self.review_df[self.dataFeature4])
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
        for line in self.review_df[self.dataFeature4]:
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
        for i in self.review_df.commentText.values:
            try:
                analysis = TextBlob(i)
                pol.append(round(analysis.sentiment.polarity,2))
            except:
                pol.append(0)

        for i in self.review_df.commentText.values:
            try:
                analysis = TextBlob(i)
                sub.append(round(analysis.sentiment.subjectivity,2))
            except:
                sub.append(0)
        self.review_df['polarity']=pol
        self.review_df['subjectivity']=sub
        self.review_df.loc[self.review_df['polarity'] < 0, 'sentimentBucket'] = -1
        self.review_df.loc[self.review_df['polarity'] == 0, 'sentimentBucket'] = 0
        self.review_df.loc[self.review_df['polarity'] > 0, 'sentimentBucket'] = 1
        self.review_df = self.review_df.loc[self.review_df['sentimentBucket'].astype(float) == float(self.sentiment)]
        #self.review_df .to_csv('youTubeVideosSentimentAnalysisSample10000.csv',sep=',',encoding='utf-8')
        #print(self.review_df )
        ##                    videoID       categoryID  views  ...    replies  polarity   subjectivity
        ##          251449  LLGENw4C1jk          17   1002386  ...      0.0      0.50          0.50
        ##          39834   3VVnY86ulA8          22    802134  ...      0.0      0.00          0.10
        ##          203460  iA86imHKCMw          17   3005399  ...      0.0     -0.08          0.69
        ##          345225  RRkdV_xmYOI          23    367544  ...      0.0      0.13          0.76
        ##          402953  vQ3XgMKAgxc          10  51204658  ...      0.0      0.25          0.50
        

    def distPlotter(self):
        self.sentimentAnalysis()
        name1 = str('commentCount')
        field = self.review_df [name]
        print(round(field.drop_duplicates().describe(include='all')),2)
        print('Kurtosis:',round(kurtosis(field),2))
        print('Pearson R Correlation Views/Comments:',pearsonr(self.review_df ['polarity'],self.review_df ['subjectivity']))
        #print('Pearson R Correlation Views/Likes:',pearsonr(self.review_df ['views'],self.review_df ['likes']))
        #print('Pearson R Correlation Views/Dislikes:',pearsonr(self.review_df ['views'],self.review_df ['dislikes']))
        plt.grid(axis='y', alpha=0.50)
        plt.title('Histogram of '+name1)
        plt.xlabel(name2)
        plt.ylabel('Subjectivity')
        plt.hist(field,bins=70)
        plt.ticklabel_format(style='plain')
        #sns.distplot(self.review_df ['views'],hist=True,fit=norm,kde=False,norm_hist=False)
        #x,y = sns.kdeplot(self.review_df ['views']).get_lines()[0].get_data()
        plt.show()

        
    def dataModify(self):
        self.sentimentAnalysis()
        self.review_df  = self.review_df[[self.dataFeature1,self.dataFeature2,self.dataFeature4,'polarity','subjectivity','sentimentBucket']]
        self.X = self.review_df.iloc[:,[self.review_df.columns.get_loc('polarity'),self.review_df.columns.get_loc('subjectivity')]].values

      
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
        self.kmeans = KMeans(self.number_clusters, init = 'k-means++',max_iter=300,n_init=10)
        self.y_kmeans = self.kmeans.fit_predict(self.X)
        self.review_df['clusters'] = self.y_kmeans
       

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


    def dataReturned(self):
        self.kMeansClustering()
        dataComm = self.review_df.loc[self.review_df['clusters'] == self.cluster]
        # dont do more than 10 comments in a sample, very computationally intensive
        dataComm.to_csv('dataComm.csv')
        comments = dataComm['commentText'].sample(10)
        self.random = '"'+comments+'"'


    def wordCloudVisualizer(self):
        self.dataReturned()
        wordcloud = WordCloud(
            background_color='white',
            stopwords= ["dtype",'commentText','object','video','Name']+stopwords.words('english'),
            max_words=200,
            max_font_size=40, 
            scale=3
            ).generate(str(self.random))
        fig = plt.figure(1, figsize=(6,6))
        plt.title('Word cloud of chosen cluster')
        plt.axis('off')
        plt.imshow(wordcloud)
        plt.show()


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


