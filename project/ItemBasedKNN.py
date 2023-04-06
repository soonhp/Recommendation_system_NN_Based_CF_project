#%%
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm
from math import isnan, isinf
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import time
import math
import os


#%%
data_name='MovieLens100K'
sim_name = 'pcc'


#%%
#Data Load and Preprocessing
if data_name=='MovieLens100K':
    data = pd.read_csv('C:/RecoSys/Data/u.data', sep='\t', names=['uid','iid','r','ts'], encoding='latin-1')
    
    item_cols=['movie id','movie title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
    item = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=item_cols,encoding='latin-1')
    item=np.array(item.drop(columns=['movie id','movie title', 'release date', 'video release date', 'IMDb URL', 'unknown']))
    # uid, iid minus one. 
    data['uid'] = np.array(data.uid) - 1
    data['iid'] = np.array(data.iid) - 1

elif data_name=='MovieLens1M':
    data = pd.read_csv(r'D:\OneDrive - 서울과학기술대학교\바탕 화면\21.1\추천시스템공부\Movielens Data\ml-1M/ratings.dat', names=['uid','iid','r','ts'],sep='\::',encoding='latin-1',header=None)
    
    item = pd.read_csv(r'D:\OneDrive - 서울과학기술대학교\바탕 화면\21.1\추천시스템공부\Movielens Data\ml-1M/movies.dat',sep='::',  encoding='latin-1',header=None)
    m_d = {}
    for n, i in enumerate(item.iloc[:,0]):
        m_d[i] = n
    item.iloc[:,0] = sorted(m_d.values())
    
    i_to_n = []
    for i in range(data.shape[0]):
        i_to_n.append(m_d[data.loc[i,'iid']])
    data['iid'] = i_to_n
    
    # uid minus one. 
    data['uid'] = np.array(data.uid) - 1
    #genre matrix
    item = item.set_index(0)
    item = np.array(item.iloc[:,1].str.get_dummies(sep='|'))

#%% distance method.
def sim_cos(u,v):
    
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0] # co-rated
    if len(ind) > 0:
        up = sum(u[ind] * v[ind])
        down = norm(u[ind]) * norm(v[ind])
        cos_sim = up/down
        if not isnan(cos_sim):
            return cos_sim
        else:
            return 0
    else:
        return 0
    
def sim_pcc(u,v):
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    
    if len(ind)>1:
        u_m = np.mean(u[ind])
        v_m = np.mean(v[ind])
        pcc = np.sum((u[ind]-u_m)*(v[ind]-v_m)) / (norm(u[ind]-u_m)*norm(v[ind]-v_m)) # range(-1,1)
        if not isnan(pcc): # case: negative denominator
            return pcc
        else:
            return 0
    else:
        return 0



def sim_msd(u,v):
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    
    if len(ind)>0:
        msd_sim = 1 - np.sum((u[ind]/5-v[ind]/5)**2)/len(ind)
        if not isnan(msd_sim):
            return msd_sim
        else:
            return 0
    else:
        return 0
    

def sim_jacc(u,v):
    ind1=np.where((1*(u==0)+1*(v==0))==0)[0] # 
    ind2=np.where((1*(u==0)+1*(v==0))!=2)[0] # 
    if len(ind1)>0:
        return (len(ind1)/len(ind2))
    else:
        return 0


#%%
# Collaborative Filtering
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

# cv validation, random state, split setting.
cv = 5
rs = 35
sk = StratifiedKFold(n_splits=cv, random_state=rs, shuffle=True)

#결과 저장 데이터프레임
result_mae_rmse = pd.DataFrame(columns=['fold','k','MAE','RMSE'])
result_topN = pd.DataFrame(columns=['fold','k','Precision','Recall','F1_score'])
count = 0


for f, (trn,val) in enumerate(sk.split(data,data['uid'].values)):
    print(f'cv: {f+1}')
    trn_data = data.iloc[trn]
    val_data = data.iloc[val]


##########################################################################################
##########################################################################################
##########################################################################################

#%%
    # train dataset rating dictionary.
    data_d_trn_data = {}
    for u, i, r in zip(trn_data['uid'], trn_data['iid'], trn_data['r']):
        if i not in data_d_trn_data:
            data_d_trn_data[i] = {u:r}
        else:
            data_d_trn_data[i][u] = r
    
    # train dataset rating dictionary. user
    data_d_trn_data_u = {}
    for u, i, r in zip(trn_data['uid'], trn_data['iid'], trn_data['r']):
        if u not in data_d_trn_data_u:
            data_d_trn_data_u[u] = {i:r}
        else:
            data_d_trn_data_u[u][i] = r

    # train dataset item rating mean dictionary.
    data_d_trn_data_mean = {}
    for i in data_d_trn_data:
        data_d_trn_data_mean[i] = np.mean(list(data_d_trn_data[i].values()))
    


    n_item = item.shape[0]
    n_user = len(set(data['uid']))

    # train rating matrix
    rating_matrix = np.zeros((n_user, n_item))
    for u, i, r in zip(trn_data['uid'], trn_data['iid'], trn_data['r']):
        rating_matrix[u,i] = r

    # test rating matrix
    rating_matrix_test = np.zeros((n_user, n_item))
    for u, i, r in zip(val_data['uid'], val_data['iid'], val_data['r']):
        rating_matrix_test[u,i] = r
        
    # test data mean
    data_d_tst_data = {}
    data_d_tst_data_mean = {}
    
    for u, i, r in zip(val_data['uid'], val_data['iid'], val_data['r']):
        if u not in data_d_tst_data:
            data_d_tst_data[u] = {i:r}
        else:
            data_d_tst_data[u][i] = r
    
    for u in data_d_tst_data:
        data_d_tst_data_mean[u] = np.mean(list(data_d_tst_data[u].values()))

#%%
#유사도계산#############################################################################################      
    print('\n')
    print(f'similarity calculation: {sim_name}')

    if sim_name=='cos':    
        sim=pdist(rating_matrix.T,metric=sim_cos)
        sim=squareform(sim)
    elif sim_name=='pcc':    
        sim=pdist(rating_matrix.T,metric=sim_pcc)
        sim=squareform(sim)
    elif sim_name=='jacc':    
        sim=pdist(rating_matrix.T,metric=sim_jacc)
        sim=squareform(sim)  
    elif sim_name=='msd':    
        sim=pdist(rating_matrix.T,metric=sim_msd)
        sim=squareform(sim) 
    
    np.fill_diagonal(sim,-1) # 정렬하기 전..! -1값 다른거로 바꿔도 될듯!?
    nb_ind=np.argsort(sim,axis=1)[:,::-1] # nearest neighbor sort, 1행 : item 0의  index,, 즉 -> 이 방향
    sel_nn=nb_ind[:,:100] # 상위100명
    sel_sim=np.sort(sim,axis=1)[:,::-1][:,:100]

    

    print('\n')
    print('prediction: k=10,20, ..., 100')
    rating_matrix_prediction = rating_matrix.copy()
        
    s=time.time()
    e=0
    for k in tqdm([10,20,30,40,50,60,70,80,90,100]):
        
        for user in range(rating_matrix.shape[0]): # user를 돌고, # user=0
            
            for p_item in list(np.where(rating_matrix_test[user,:]!=0)[0]): # 예측할 item을 돌고, p_item=252
                
                molecule = []
                denominator = []
                
                # call K neighbors 아이템 p_item 이랑 유사한 k개의 아이템들이 item_neihbor이 되고,,
                item_neighbor = sel_nn[p_item,:k]
                item_neighbor_sim = sel_sim[p_item,:k]

                for neighbor, neighbor_sim in zip(item_neighbor, item_neighbor_sim): # neighbor=337
                    if neighbor in data_d_trn_data_u[user].keys():
                        molecule.append(neighbor_sim * (rating_matrix[user, neighbor] - data_d_trn_data_mean[neighbor]))
                        denominator.append(abs(neighbor_sim))
                try:
                    rating_matrix_prediction[user, p_item] = data_d_trn_data_mean[p_item] + (sum(molecule) / sum(denominator))
                except : #ZeroDivisionError: user가 p_item의 이웃item을 평가한 적이 없는 경우, KeyError: test에는 있는데 train에는 없는 item.
                    e+=1
                    rating_matrix_prediction[user, p_item] = math.nan
                  
          #3. performance
        # MAE, RMSE
        
        precision, recall, f1_score = [], [], []
        # 평균!!
    
        pp=[]
        rr=[]
        n_tp = 0
        n_fp = 0
        n_fn = 0
        for u, i, r in zip(val_data['uid'], val_data['iid'], val_data['r']):
            p = rating_matrix_prediction[u,i]
            if not math.isnan(p):
                pp.append(p) # 예측
                rr.append(r) # 실제
                u_mean = data_d_tst_data_mean[u]
                if p >= u_mean and r >= u_mean:
                    n_tp += 1
                elif p >= u_mean and r < u_mean:
                    n_fp += 1
                elif p < u_mean and r >= u_mean:
                    n_fn += 1
        _precision = n_tp / (n_tp + n_fp)
        _recall = n_tp / (n_tp + n_fn)
        _f1_score = 2 * _precision * _recall / (_precision + _recall)      
                
    
        d = [abs(a-b) for a,b in zip(pp,rr)]
        mae = sum(d)/len(d)
        rmse = np.sqrt(sum(np.square(np.array(d)))/len(d))
        
        result_mae_rmse.loc[count] = [f, k, mae, rmse]
        result_topN.loc[count] = [f,k, _precision, _recall, _f1_score]
        count += 1

result_1 = result_mae_rmse.groupby(['k']).mean().drop(columns=['fold'])
result_2 = result_topN.groupby(['k']).mean().drop(columns=['fold'])
result = pd.merge(result_1, result_2, on=result_1.index).drop(columns=['key_0'])
print(result)

result.to_csv('result/'+sim_name+'.csv')
    

    
