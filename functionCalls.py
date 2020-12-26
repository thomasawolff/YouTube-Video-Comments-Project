from sentenceVectConnect2 import *
import sys

url = (r'C:\Users\moose_f8sa3n2\Google Drive\Research Methods\Course Project\YouTube Data\Unicode Files\youTubeVideosUTF.csv')



categories = ['Film & Animation', 'Autos & Vehicles', 'Music', 'Pets & Animals', 'Sports'
          , 'Short Movies', 'Travel & Events', 'Gaming', 'Videoblogging', 'People & Blogs'
          , 'Comedy', 'Entertainment', 'News & Politics', 'Howto & Style', 'Education'
          , 'Science & Technology', 'Nonprofits & Activism', 'Movies', 'Anime/Animation'
          , 'Action/Adventure', 'Classics', 'Comedy', 'Documentary', 'Drama', 'Family'
          , 'Foreign', 'Horror', 'Sci-Fi/Fantasy', 'Thriller', 'Shorts', 'Shows', 'Trailers']

print('')
print('')
##print(categories)
##print('')
##print('')

while True:
    exit_ = input('Type Done to leave or press enter ')
    if exit_.lower() == 'done': sys.exit(0)
    clusters = input('Do you know how many clusters you want? Yes/No: ')
    if clusters.lower() == 'yes':
        try:
            try:
                go = textAnalytics(url,
                                   numClusters = int(input('How many clusters do you want?: ')),
                                   channel = input('Which channel do you want to analyze?: '),
                                   #category = input('Which category do you want to analyze?: '),
                                   dataFeature1 = 'videoID', # First of Four columns in dataset
                                   dataFeature2 = 'categoryID',
                                   #dataFeature3 = 'views',
                                   dataFeature4 = 'commentText',
                                   sentiment = input('Which sentiment do you want? (1.0 for positive,0.0 for nuetral,-1.0 for negative): '))

                #go.kMeansElbow()
                go.kMeansVisualizer()
                go.triGramConverter()
                go.tagsMaker()
                go.wordCloudVisualizer()
                go.plot_similarity()
            except ValueError:
                print('You may have entered bad data')
                pass
        except TypeError:
            print('You Entered an invalid cluster number, try: '+str(int(go.cluster)-1))
            pass
    else:
         try:
            try:
                 go = textAnalytics(url,
                                   #category = input('Which category do you want to analyze?: '),
                                   channel = input('Which channel do you want to analyze?: '),
                                   dataFeature1 = 'videoID', # First of Four columns in dataset
                                   dataFeature2 = 'categoryID',
                                   dataFeature3 = 'views',
                                   dataFeature4 = 'commentText',
                                   sentiment = input('Which sentiment do you want? (1.0 for positive,0.0 for nuetral,-1.0 for negative): '))

                 go.kMeansElbow()
            except ValueError:
                print('You may have entered bad data')
                pass
         except TypeError:
            print('You Entered an invalid cluster number, try: '+str(int(go.cluster)-1))
            pass
