import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#计算余弦相似度
def Cosine(list1,list2):
    res = 0
    d1 = 0
    d2 = 0
    for(v1,v2) in zip(list1,list2):
        res += (v1 * v2)
        d1 += (v1 ** 2)
        d2 += (v2 ** 2)

    return res / math.sqrt(d1 * d2)

if __name__ == '__main__':
    #读入数据
    ratings = pd.read_csv('./data/ratings.csv', index_col=None)
    movies = pd.read_csv('./data/movies.csv', index_col=None)
    #print(ratings.describe())
    #print(ratings.head(5))

    #构建透视表
    ratingsPivo = pd.pivot_table(ratings[['userId', 'movieId', 'rating']], columns=['movieId'],
                                 index=['userId'], values='rating', fill_value=0)

    #movie id表 & user id表 & 评分表
    movieId = dict(enumerate(list(ratingsPivo.columns)))
    userId = dict(enumerate(list(ratingsPivo.index)))
    ratingValue = ratingsPivo.values.tolist()
    #print(trainRatinsPivo)
    #print(movieId)
    #print(userId)
    #print(ratingValue)

    #计算用户相似度
    userSim = np.zeros((len(ratingValue),len(ratingValue)),dtype=np.float32)
    for i in range(len(ratingValue)-1):
        for j in range(i+1,len(ratingValue)):
            userSim[i,j] = Cosine(ratingValue[i],ratingValue[j])
            userSim[j,i] = userSim[i,j]

    #print(userSim)

    #选取前10个最相似的用户
    userMostSim = dict()
    for i in range(len(ratingValue)):
        userMostSim[i] = sorted(enumerate(list(userSim[i])),key = lambda x:x[1],reverse=True)[:10]

    #计算用户对每部电影的兴趣分
    userRecValue = np.zeros((len(ratingValue),len(ratingValue[0])),dtype=np.float32)
    for i in range(len(ratingValue)):
        for j in range(len(ratingValue[i])):
            if ratingValue[i][j] == 0:
                v = 0
                for (user,sim) in userMostSim[i]:
                    v += (ratingValue[user][j] * sim)
                userRecValue[i,j] = v

    #选取前3个最感兴趣的电影
    userRecDict = dict()
    for i in range(len(ratingValue)):
        userRecDict[i] = sorted(enumerate(list(userRecValue[i])),key = lambda x:x[1],reverse=True)[:3]

    #整理输出
    userRecList = []
    for key,value in userRecDict.items():
        user = userId[key]
        for(movie,val) in value:
            userRecList.append([user,movieId[movie]])

    rec = pd.DataFrame(userRecList,columns=['userId','movieId'])
    rec.to_csv('./data/movie.csv',index = False)
    print("Complete!!!")


