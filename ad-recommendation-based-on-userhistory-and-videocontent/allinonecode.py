#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


# In[3]:


#Read database and add index column if necessary
asmidf = pd.read_csv("datasetasmi/uploadedVideosEdited.csv")
# print(asmidf.shape)
# print(len(asmidf))
# asmidf['index']=range(1, len(asmidf) + 1)
asmidf.head()


# In[4]:


classes = ["Shirt","Trousers","Footwear","Handbag","Watch","Guitar","Mobile_phone","Headphones","Hat","Sunglasses"]

#Create tuple list if necessary
# alltupleList=[]
# for i in range(0,len(asmidf)):
#     detectedobjectsstr=str(asmidf.iloc[i]['detected_objects_withconfidence'])
#     detectedobjectswithScore=detectedobjectsstr.split("|")

#     for eachobjectwithScore in detectedobjectswithScore:
#         if eachobjectwithScore.split(":")[0] in classes:
#             #( index that starts from 0 , class , score )
#             alltupleList.append( ( i , eachobjectwithScore.split(":")[0],int(eachobjectwithScore.split(":")[1])  ) )

# print(alltupleList)


# In[5]:


#Generate a new dataframe with all the scores of detected objects

currentindex=0
def class_score(row):
    detectedobjectsstr=str(row['detected_objects_withconfidence'])
    detectedobjectswithScore=detectedobjectsstr.split("|")
    currentclass=classes[currentindex]
    
    for eachobjectwithScore in detectedobjectswithScore:
        if eachobjectwithScore.split(":")[0] in classes and eachobjectwithScore.split(":")[0]==currentclass:
            return int(int(eachobjectwithScore.split(":")[1]))

# newdf=asmidf[['video_id']].copy()
## We have not made video_id as one of the columns because video_id, which is not a feature,will also be considered 
## and used to calculate similarity matrix 
newdf = pd.DataFrame(columns = None)
for eachclass in classes:
    currentindex=classes.index(eachclass)
    newdf[eachclass]=asmidf.apply(class_score,axis=1)

# print(newdf.shape)
newdf.head()

# df.head()


# In[6]:


from sklearn.metrics import pairwise_distances


#Use Cosine, if data is sparse (many ratings are undefined)
# metric="cosine"

# We have choosen to use Euclidean metric because the sparsity of our data is not that high
# and the magnitude of the attribute values is significant
# Later we will change it to cosine because our AI algorithms might produce comparatively more sparse matrix
metric="euclidean"

similarity_max = 1-pairwise_distances(newdf, metric=metric)
pd.DataFrame(similarity_max)


# In[27]:


#This function finds k similar video given the video_id and ratings matrix M
#Note that the similarities are same as obtained via using pairwise_distances
k=1
from sklearn.neighbors import NearestNeighbors

def getvideoindexfromvideoid(video_id,matrixwithvideoid):
#     for j in range(len(feature_matrix)):
    for j in range(matrixwithvideoid.shape[0]):
        if matrixwithvideoid.iloc[j]['video_id'] == video_id:
            return j
    return -1

def getvideoidfromindex(index,matrixwithvideoid):
    if index<=matrixwithvideoid.shape[0]:
        return matrixwithvideoid.iloc[index]['video_id']
    else:
        return -1

def findksimilarvideo(video_id, feature_matrix,matrixwithvideoid, metric = "euclidean", k=k):
    similarities=[]
    indices=[]
    finalvideoIds=[]
    videoindex=getvideoindexfromvideoid(video_id,matrixwithvideoid)
    
#     print("video_id: {}".format(video_id))
    print("videoindex: {}".format(videoindex))
    if videoindex==-1:
        print("Row with the video_id {} wasnot found.".format(video_id))
        return finalvideoIds
    
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
    model_knn.fit(feature_matrix)
    
    distances, indices = model_knn.kneighbors(feature_matrix.iloc[videoindex, :].values.reshape(1, -1), n_neighbors = k+1)
#     print(distances.flatten())
    similarities = 1-distances.flatten()
#     print("Similarity coefficient : ",similarities)
#     print("Index for similarity   : ",indices.flatten())
    
#     print('{0} most similar videos for Video {1}:\n'.format(k,video_id))
    
    
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == videoindex:
            continue;
        else:
            requiredvideoId = getvideoidfromindex(indices.flatten()[i],matrixwithvideoid)
            finalvideoIds.append(requiredvideoId)
#             print('video_index:{0} video_id:{1} with similarity of {2}'.format( indices.flatten()[i],requiredvideoId,similarities.flatten()[i]))
            
    return finalvideoIds


# In[29]:



# allvideoIds = findksimilarvideo(48,newdf,asmidf, metric='euclidean',k=4)
# print(allvideoIds)

allvideoIds = findksimilarvideo(5336,newdf,asmidf, metric='euclidean',k=2)
print(allvideoIds)
allvideoIds = findksimilarvideo(7828,newdf,asmidf, metric='euclidean',k=2)
print(allvideoIds)

newdf.iloc[5290, :].values.reshape(1, -1)


# In[ ]:





# In[9]:


###################################Find which video was significant for the user#########################

#Read database and add index column if necessary
view_history_raw_df = pd.read_csv("datasetasmi/userViewHistories.csv")
# print(asmidf.shape)
# print(len(asmidf))
# asmidf['index']=range(1, len(asmidf) + 1)
view_history_raw_df.head()


# In[10]:


#Generate new dataframe with WatchCount(watch_count)

# sorted_view_history_rawdf = view_history_raw_df.sort_values(by=['watched_video_id'], ascending=True)
# print(sorted_view_history_rawdf)

unique_user_id_list =view_history_raw_df.user_id.unique()
unique_video_id_list =view_history_raw_df.watched_video_id.unique()
# print(len(unique_user_id_list))
# print(len(unique_video_id_list))

counter=0
# featured_view_history = pd.DataFrame(columns = ['user_id','watched_video_id','watch_count','total_watch_time'])

allrowsList=[]
for eachvideoid in unique_video_id_list:
    for eachuserid in unique_user_id_list:
        #Check conditions
        rowsthatsatisfy = view_history_raw_df[(view_history_raw_df['watched_video_id']==eachvideoid) & (view_history_raw_df['user_id']==eachuserid)]
#         print(rowsthatsatisfy)
        count=len(rowsthatsatisfy)
#         print(count)
        if count==0:
            continue
        totalwatchtime=rowsthatsatisfy['watch_time_in_sec'].sum()
        rowdict={'user_id':eachuserid,'watched_video_id':eachvideoid,'watch_count':count,'total_watch_time':totalwatchtime}
        allrowsList.append(rowdict.copy())
featured_view_history = pd.DataFrame(allrowsList)
# featured_view_history.head()
print(featured_view_history.shape)

# featured_view_history = featured_view_history.sort_values(by=['watched_video_id'], ascending=True)
featured_view_history.head()


# In[11]:


#Normalise the data according to average
word_count_avg=featured_view_history['watch_count'].mean()
watch_time_avg=featured_view_history['total_watch_time'].mean()
print("Average watch count is : ",word_count_avg)
print("Average watch time is : ",watch_time_avg)

currentindex=0
current_class_name=''
def avg_out_score(row):
    return row[current_class_name]/avgList_forcolumns[currentindex]

columnsToBeNormalised=['watch_count','total_watch_time']
avgList_forcolumns = [word_count_avg,watch_time_avg]

normalisedf=featured_view_history[['user_id','watched_video_id','total_watch_time']].copy()

# normaliseddf = pd.DataFrame(columns = None)
for eachclass in columnsToBeNormalised:
    current_class_name=eachclass
    currentindex=columnsToBeNormalised.index(eachclass)
    normalisedf['avg_'+eachclass]=featured_view_history.apply(avg_out_score,axis=1)

watch_time_importance = 0.7
watch_count_importance = 0.3

def calculate_video_importance(row):
    return row['avg_watch_count']*watch_count_importance*row['avg_total_watch_time']*watch_time_importance

normalisedf['video_importance']=normalisedf.apply(calculate_video_importance,axis=1)

# normalisedf = normalisedf.sort_values(by=['video_importance'], ascending=False)
normalisedf.head()


# In[12]:


##Find most important video
number_of_video=1

def findmostimportantvideo(user_id,dataframe,number_of_video=number_of_video):
    dataframe=dataframe[dataframe['user_id']==user_id]
    dataframe=dataframe.sort_values(by=['video_importance'], ascending=False)
    return dataframe.head(number_of_video)['watched_video_id'].values


# In[13]:


requiredVideoIDList = findmostimportantvideo(1,normalisedf,number_of_video=2)
print(requiredVideoIDList)


# In[14]:


############################################# Integrating #####################################################
current_user_id=1
current_beingwatched_video_id=48
number_of_history_based_imp_videoid = 1

# find important videos according to who the user is and his views history 
requiredVideoIDList = findmostimportantvideo(current_user_id,normalisedf,number_of_video=number_of_history_based_imp_videoid)
print(requiredVideoIDList)


# In[15]:


# Make a list of similar videos including important videos

similar_videos_list=[]
number_of_similar_videos=1

for i in range(len(requiredVideoIDList)):
    current_video_id = requiredVideoIDList[i]
    similar_videos_list.append(current_video_id)
    allvideoIds = findksimilarvideo(current_video_id,newdf,asmidf, metric='euclidean',k=number_of_similar_videos)
    for j in range(len(allvideoIds)):
        similar_videos_list.append(allvideoIds[j])

# If I had to recommend new video...I would recommend these without appending current_video_id as they had been already watched
print(similar_videos_list)


# In[26]:


#### Extract important features from the videoid
## First use already available newdf for 'FEATURES' and asmidf for "VIDEO_ID" to make matrix_with_bothFeatureAndVideoId

matrix_with_bothFeatureAndVideoId = newdf.copy()
matrix_with_bothFeatureAndVideoId['video_id']= asmidf['video_id']
matrix_with_bothFeatureAndVideoId.head()


# In[17]:



# matrix_with_bothFeatureAndVideoId=matrix_with_bothFeatureAndVideoId[matrix_with_bothFeatureAndVideoId['video_id'] in similar_videos_list]
filtered_df = matrix_with_bothFeatureAndVideoId.loc[matrix_with_bothFeatureAndVideoId['video_id'].isin(similar_videos_list)]
filtered_df.head()


# In[18]:


listoffeatures = filtered_df.columns.tolist()
listoffeatures.remove('video_id')


# In[19]:


##### MAIN SCORES TUPLE FROM USER HISTORY
features_withscores_from_userhistory = []

def make_required_tuple_from_df(listoffeatures,df=filtered_df):
    features_withscores=[]
    for i in range(len(listoffeatures)):
        currentfeature = listoffeatures[i]
        currentfeaturescore = df[currentfeature].mean()
        features_withscores.append((currentfeature,currentfeaturescore))
    return features_withscores

    


# In[20]:


features_withscores_from_userhistory=make_required_tuple_from_df(listoffeatures,df=filtered_df)
print(features_withscores_from_userhistory)


# In[21]:


######## LETS START EXTRACTING AVAILABLE FEATURE FROM CURRENT VIDEO THAT USER IS WATCHING

filtered_df = matrix_with_bothFeatureAndVideoId.loc[matrix_with_bothFeatureAndVideoId['video_id']==current_beingwatched_video_id]

listoffeatures = filtered_df.columns.tolist()
listoffeatures.remove('video_id')
listoffeatures


# In[22]:


## Make tuple as above:
features_withscores_from_videocontent =make_required_tuple_from_df(listoffeatures,df=filtered_df)
print(features_withscores_from_videocontent)


# In[23]:


### Till now
# From userhistory,  we got  features_withscores_from_userhistory
# From videocontent, we got  features_withscores_from_videocontent

significance_of_userhistory  = 0.5
significance_of_videocontent = 0.5

def getindexfromTupleListbasedonfeature(featurename,tuplename):
#     for j in range(len(feature_matrix)):
    for j in range(len(tuplename)):
        if tuplename[j][0] == featurename:
            return j
    return -1

avgout_feature_score=[]
for i in range(len(listoffeatures)):
    currentfeature = listoffeatures[i]
    valuefrom_userhistory = features_withscores_from_userhistory[getindexfromTupleListbasedonfeature(currentfeature,features_withscores_from_userhistory)][1]
    valuefrom_videocontent = features_withscores_from_videocontent[getindexfromTupleListbasedonfeature(currentfeature,features_withscores_from_videocontent)][1]
    average = valuefrom_userhistory * significance_of_userhistory + valuefrom_videocontent * significance_of_videocontent
    avgout_feature_score.append((currentfeature,average))

print(avgout_feature_score)

sorted_avgout_feature_score = sorted(avgout_feature_score, key=lambda k: k[1],reverse=True)
print(sorted_avgout_feature_score)


# In[24]:


## Function that gets top scoring ads

def getNamesofTopScoringAdsFromTupleOfAverage(numberofads,avg_tuple):
    adnames = []
    for i in range(numberofads):
        adnames.append(avg_tuple[i][0])
    return adnames
    

numberofadstobeshown = 2
nameofads = getNamesofTopScoringAdsFromTupleOfAverage(numberofadstobeshown,avgout_feature_score)
print(nameofads)


# In[ ]:





# In[ ]:




