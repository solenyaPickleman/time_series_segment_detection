import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hac 
import gensim
import datetime
import operator 

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from functools import reduce
from gensim.similarities import MatrixSimilarity
from collections import Counter
from matplotlib import pyplot as plt


#read in data
dat = pd.read_csv('/home/sir/projects/time_series_segment_detection/data.csv')

#break data into windows

dat['date'] = dat['date'].apply(pd.to_datetime)
dat = dat.sort_values(by=['date'], ascending=True)
dat = dat[3000:4000]

mindate = dat['date'].iloc[0]
maxdate = dat['date'].iloc[-1]
numdays = (maxdate-mindate).days+1

window = 7
#build windows
dateranges = []
for offset in range(0, numdays) : 
    rangestart = datetime.timedelta(days=offset)
    rangeend = datetime.timedelta(days=offset+window)
    dateranges.append((mindate+rangestart, mindate+rangeend))
    
#using the dates as indices 
dat = dat.set_index('date')
#get words in each window
windows = []

for start, end in dateranges:
    values = dat[start:end]
    values = reduce (operator.add, values.values.tolist())
    windows.append(values)

#troublesome...... or at least - will need to tie dates back to clusters
windows = [x for x in windows if len(x) != 0]

#split common

#build TFIDF 
dct = Dictionary(windows)
corpus = [dct.doc2bow(line) for line in windows]

sim = MatrixSimilarity(corpus, num_features=len(dct))
sim = sim[corpus]

#encourage detection of static processes by?
#Initial : if there is a record with a nonzero min similarity
# then it is more likely that the data is representing a static process
static_indicator = np.sum([np.min(x) for x in sim])

#can i be lazy? scipy time
#TODO:  improve w/ heuristic being 

linked = hac.linkage(sim, method='single', metric='euclidean')

#use the static indicator to "suggest" that there is a static process. 
threshold = 0.05 if static_indicator == 0.0 else  1.0
thresholdIncrement = 0.05
clusters = hac.fcluster( linked ,t=threshold ,criterion='distance')

#set threshold to less then window
num_singletons = len([True for cl, count in Counter(clusters).items() if count ==  1])
while num_singletons > 0 : 
    threshold += thresholdIncrement
    clusters = hac.fcluster( linked ,t=threshold ,criterion='distance')
    num_singletons = len([True for cl, count in Counter(clusters).items() if count == 1])
    print (threshold, num_singletons)

plt.figure(figsize=(10,7))
hac.dendrogram(linked, color_threshold=threshold, orientation='top', distance_sort='descending') 
plt.savefig('/mnt/c/Users/Sir/Documents/fig.png')

# for x,y in zip(clusters, windows) : 
#     print(x, sorted(y))
#tie index into dates