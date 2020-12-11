from sentenceVectConnect2 import *

url = (r'C:\Users\moose_f8sa3n2\Google Drive\Research Methods\Course Project\YouTube Data\Unicode Files\youTubeVideosUTF.csv')

go = textAnalytics(url,numClusters=4,cluster=3,category='Music')
#go.kMeansElbow()
go.kMeansVisualizer()
go.wordCloudVisualizer()
go.plot_similarity()
