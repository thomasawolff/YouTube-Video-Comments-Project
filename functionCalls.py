from sentenceVectDBConnect import *


go = commentVectors(3,10000,10) # calling plot_similarity() method for K means cluster 3, 
                                # over 10,000 samples from textAnalytics class, 
                                # sampling 10 comments from those 10,000 samples
if __name__ == '__main__':
    Pool(go.plot_similarity())


#modelPredictionsLR()
