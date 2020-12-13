from sentenceVectConnect2 import *

url = (r'C:\Users\moose_f8sa3n2\Google Drive\Research Methods\Course Project\YouTube Data\Unicode Files\youTubeVideosUTF.csv')



categories = ['Film & Animation', 'Autos & Vehicles', 'Music', 'Pets & Animals', 'Sports'
          , 'Short Movies', 'Travel & Events', 'Gaming', 'Videoblogging', 'People & Blogs'
          , 'Comedy', 'Entertainment', 'News & Politics', 'Howto & Style', 'Education'
          , 'Science & Technology', 'Nonprofits & Activism', 'Movies', 'Anime/Animation'
          , 'Action/Adventure', 'Classics', 'Comedy', 'Documentary', 'Drama', 'Family'
          , 'Foreign', 'Horror', 'Sci-Fi/Fantasy', 'Thriller', 'Shorts', 'Shows', 'Trailers']

go = textAnalytics(url,
                   numClusters=3, # How many clusters do you want?
                   KmeansColumn1=3, # First column for K means clustering
                   KmeansColumn2=4, # Second column for K means clustering
                   cluster=2, # Which cluster do you want to analyze?
                   category='Pets & Animals', # Which category do you want to analyze?
                   dataFeature1 = 'videoID', # First of Four columns in dataset
                   dataFeature2 = 'categoryID',
                   dataFeature3 = 'views',
                   dataFeature4 = 'commentText',
                   sentiment = 1.0) # 1.0 for positive, 0.0 for nuetral, -1.0 for negative
#go.kMeansElbow()
go.dendrogram()
#go.kMeansVisualizer()
#go.wordCloudVisualizer()
#go.plot_similarity()
