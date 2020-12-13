from sentenceVectConnect2 import *

url = (r'C:\Users\moose_f8sa3n2\Google Drive\Research Methods\Course Project\YouTube Data\Unicode Files\youTubeVideosUTF.csv')



categories = ['Film & Animation', 'Autos & Vehicles', 'Music', 'Pets & Animals', 'Sports'
          , 'Short Movies', 'Travel & Events', 'Gaming', 'Videoblogging', 'People & Blogs'
          , 'Comedy', 'Entertainment', 'News & Politics', 'Howto & Style', 'Education'
          , 'Science & Technology', 'Nonprofits & Activism', 'Movies', 'Anime/Animation'
          , 'Action/Adventure', 'Classics', 'Comedy', 'Documentary', 'Drama', 'Family'
          , 'Foreign', 'Horror', 'Sci-Fi/Fantasy', 'Thriller', 'Shorts', 'Shows', 'Trailers']

# 'polarity','subjectivity'
print('')
print('')
print(categories)
print('')
print('')
go = textAnalytics(url,
                   numClusters= input('How many clusters do you want? '),
                   cluster= input('Which cluster do you want to analyze? '),
                   category=input('Which category do you want to analyze? '),
                   dataFeature1 = 'videoID', # First of Four columns in dataset
                   dataFeature2 = 'categoryID',
                   dataFeature3 = 'views',
                   dataFeature4 = 'commentText',
                   sentiment = input('Which sentiment do you want? (1.0 for positive, 0.0 for nuetral, -1.0 for negative) '))

print(go.number_clusters)

if go.number_clusters == '':
    go.kMeansElbow()
##else:
##    go.kMeansElbow()
##    go.kMeansVisualizer()
##    go.wordCloudVisualizer()
##    go.plot_similarity()

input('Press Enter to leave')
