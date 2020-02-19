import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

#Read database and add index column if necessary
asmidf = pd.read_csv("datasetasmi/uploadedVideosEdited.csv")
# print(asmidf.shape)
# print(len(asmidf))
# asmidf['index']=range(1, len(asmidf) + 1)
asmidf.head()


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




#Generate a new dataframe with all the scores of detected objects
currentindex=0
def class_score(row):
    detectedobjectsstr=str(row['detected_objects_withconfidence'])
    detectedobjectswithScore=detectedobjectsstr.split("|")
    currentclass=classes[currentindex]
    
    for eachobjectwithScore in detectedobjectswithScore:
        if eachobjectwithScore.split(":")[0] in classes and eachobjectwithScore.split(":")[0]==currentclass:
            return int(int(eachobjectwithScore.split(":")[1]))

newdf=asmidf[['video_id']].copy()
for eachclass in classes:
    currentindex=classes.index(eachclass)
    newdf[eachclass]=asmidf.apply(class_score,axis=1)

# print(newdf.shape)
newdf.head()