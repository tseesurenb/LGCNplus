'''
Created on Oct 12, 2023
Pytorch Implementation of TempLGCN in
Tseesuren et al. tempLGCN: Simple and Time-aware Graph Convolution Network for Collaborative Filtering Recommendation

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

from surprise import Reader
from surprise import Dataset
from sklearn import preprocessing
import pandas as pd
#import math
import torch
from torch_sparse import SparseTensor
import numpy as np
#import datetime
from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle

import pandas as pd
import matplotlib.pyplot as plt

# load dataset        
def load_data(dataset = "ml100k", verbose = False):
    
    if dataset == "ml1m": #ml1m dataset
        _reader = Reader(line_format="user item rating timestamp", sep="::") 
        _data = Dataset.load_from_file("data/ml-1m/ratings.dat", reader=_reader)
    elif dataset == "ml10m": #ml10m dataste
        _reader = Reader(line_format="user item rating timestamp", sep="::")
        _data = Dataset.load_from_file("data/ml-10m/ratings.dat", reader=_reader)
    elif dataset == "ml100k": #ml100k dataset
        _reader = Reader(line_format="user item rating timestamp", sep="\t")
        _data = Dataset.load_from_file("data/ml-100k/u.data", reader=_reader)
    elif dataset == "dummy": #dummy dataset
        _reader = Reader(line_format="user item rating timestamp", sep="\t") 
        _data = Dataset.load_from_file("data/dummy/test.data", reader=_reader)
    elif dataset == "amazon": #dummy dataset
        _reader = Reader(line_format="user item rating timestamp", sep="\t") 
        _data = Dataset.load_from_file("data/amazon/df_modcloth.csv", reader=_reader)
        #item_id,user_id,rating,timestamp,size,fit,user_attr,model_attr,category,brand,year,split
        
    rating_df = pd.DataFrame(_data.raw_ratings, columns=['userId', 'itemId', 'rating', 'timestamp'])    
    _lbl_user = preprocessing.LabelEncoder()
    _lbl_movie = preprocessing.LabelEncoder()

    rating_df.userId = _lbl_user.fit_transform(rating_df.userId.values)
    rating_df.itemId = _lbl_movie.fit_transform(rating_df.itemId.values)
        
    num_users = len(rating_df['userId'].unique())
    num_items = len(rating_df['itemId'].unique())
    mean_rating = rating_df['rating'].mean()
    
    if verbose:
        print(f"Using {dataset} dataset.")
        print("The max ID of Users:", rating_df.userId.max(), ", The max ID of Items:", rating_df.itemId.max())   
        print(f"#Users: {num_users}, #Items: {num_items}")
        
    return rating_df, num_users, num_items, mean_rating

# add time distance scaled with beta
def add_u_abs_decay(rating_df, beta = 0.25, method = 'linear', verbose = False):
    
    _beta = beta # hyperparameter that defines time distance weight
    _exp_beta = 5
    _base = 0.000000001
    _win_unit = 24*3600
    
    if verbose:
        print(f'The beta in item drift:{_beta}')
        
    rating_df["timestamp"] = rating_df["timestamp"].astype("int64")
    _start = rating_df['timestamp'].min()
    _end = rating_df['timestamp'].max()
    
    _max_distance = _end - _start 
    
    if method == 'linear':
        rating_df['u_abs_decay'] = _base + ((rating_df['timestamp'] - _start) / _max_distance)
    if method == 'log':
        rating_df['u_abs_decay'] = _base + np.power(((rating_df['timestamp'] - _start) / _max_distance), _beta)
    if method == 'log_old':
        rating_df['u_abs_decay'] = _base + np.power(((rating_df['timestamp'] - _start) / _win_unit), _beta)
    if method == 'recip':
        rating_df['u_abs_decay'] = _base + np.power(((rating_df['timestamp'] - _start) / _max_distance), 1/_beta)
    if method == 'exp':
        rating_df['u_abs_decay'] = _base + np.exp(-_beta * (rating_df['timestamp'] - _start) / _max_distance)
                
    
    print(f'The absolute decay method is {method} with param: {beta}')
    #print(rating_df['u_abs_decay'])
    num_uq_dists = len(rating_df['u_abs_decay'].unique())
    
    return num_uq_dists

# convert timestamp to day, week, month level
def add_u_pref_rel_decay(rating_df, beta = 0.25, method = 'linear', verbose = False):
    
    local_agg_emb_len = 0
    new_df = None
    _beta = beta
    _base = 0.000000001
    _win_unit = 24*3600
    
    if verbose:
        print(f'The beta in user drift:{_beta}')
        
    # Convert timestamp to int64
    rating_df["timestamp"] = rating_df["timestamp"].astype("int64")
    
    # Find the minimum and maximum timestamp for each user
    user_min_timestamp = rating_df.groupby('userId')['timestamp'].min()
    user_max_timestamp = rating_df.groupby('userId')['timestamp'].max()
    
    new_df_list = []
    local_agg_emb_len_list = []
    
    # Iterate over each user
    for user_id, (_start, _end) in zip(user_min_timestamp.index, zip(user_min_timestamp, user_max_timestamp)):
        # Calculate _dist for each user
        _max_distance = _end - _start
        
        # Calculate new timestamp for each user
        if method == 'log_old':
            rating_df.loc[rating_df['userId'] == user_id, 'u_rel_decay'] = (1 + ((rating_df['timestamp'] - _start) / _win_unit)) ** (_beta)
        elif method == 'linear':
            rating_df.loc[rating_df['userId'] == user_id, 'u_rel_decay'] = _base + ((rating_df['timestamp'] - _start) / _max_distance)
        elif method == 'log':
            rating_df.loc[rating_df['userId'] == user_id, 'u_rel_decay'] = _base + np.power(((rating_df['timestamp'] - _start) / _max_distance), _beta)
        elif method == 'recip':
            rating_df.loc[rating_df['userId'] == user_id, 'u_rel_decay'] = _base + np.power(((rating_df['timestamp'] - _start) / _max_distance), 1/_beta)
        elif method == 'exp':
            rating_df.loc[rating_df['userId'] == user_id, 'u_rel_decay'] = _base + np.exp(-_beta * (rating_df['timestamp'] - _start) / _max_distance)
        
        # Create a new DataFrame for the user
        new_user_df = rating_df.loc[rating_df['userId'] == user_id, ['userId', 'itemId', 'rating', 'timestamp', 'u_abs_decay', 'u_rel_decay']]
        
        # Append the DataFrame to the list
        new_df_list.append(new_user_df)
        
        # Calculate the length of unique new_u_ts_id values for each user
        local_agg_emb_len_list.append(len(new_user_df['u_rel_decay'].unique()))

    # Concatenate DataFrames for all users
    new_df = pd.concat(new_df_list, ignore_index=True)
    
    if verbose:
        # Get the sorted values of 'new_u_ts_id'
        sorted_values = sorted(new_df['u_rel_decay'])
        # Plot the sorted values
        plt.plot(sorted_values)
        # Add labels and title
        plt.xlabel('Index (after sorting)')
        plt.ylabel('u_rel_decay')
        plt.title('Sorted Plot of u_rel_decay')
        # Show the plot
        plt.show()
    
    # Sum of local_agg_emb_len values for all users
    local_agg_emb_len = sum(local_agg_emb_len_list)
    
    print(f'The relative decay method is {method} with param: {beta}')
    
    return new_df, local_agg_emb_len

# get user stats for each user including number of ratings, mean rating and time distance
def get_user_stats(rating_df, verbose = False):
    
    _dist_unit = 24*3600 # a day
    
    # Group by userId and calculate the required statistics
    user_stats = rating_df.groupby('userId').agg(
        num_ratings=pd.NamedAgg(column='rating', aggfunc='count'),
        mean_rating=pd.NamedAgg(column='rating', aggfunc='mean'),
        time_distance=pd.NamedAgg(column='timestamp', aggfunc=lambda x: (x.max() - x.min())/_dist_unit)
    ).reset_index()

    # Rename columns for clarity
    user_stats.columns = ['userId', 'num_ratings', 'mean_rating', 'time_distance']

    return user_stats

# get item stats for each item including number of ratings, mean rating and time distance
def get_item_stats(rating_df, verbose = False):
    
    _dist_unit = 24*3600 # a day
    
    # Group by userId and calculate the required statistics
    item_stats = rating_df.groupby('itemId').agg(
        num_ratings=pd.NamedAgg(column='rating', aggfunc='count'),
        mean_rating=pd.NamedAgg(column='rating', aggfunc='mean'),
        time_distance=pd.NamedAgg(column='timestamp', aggfunc=lambda x: (x.max() - x.min())/_dist_unit)
    ).reset_index()

    # Rename columns for clarity
    item_stats.columns = ['itemId', 'num_ratings', 'mean_rating', 'time_distance']

    return item_stats

# get edge ids and edge values between users (source) and items (dest)
def get_edge_values(rating_df, rating_threshold = 0, verbose = False):

    _src = [user_id for user_id in rating_df["userId"]]
    _dst = [item_id for item_id in rating_df["itemId"]]
    _link_vals = rating_df["rating"].values
    _ts = rating_df["timestamp"].values
    _abs_decay = rating_df["u_abs_decay"].values
    _rel_decay = rating_df["u_rel_decay"].values
    
    _true_edges = torch.from_numpy(rating_df["rating"].values).view(-1, 1).to(torch.long) >= rating_threshold
    
    edge_index = [[],[]]
    edge_values = []
    edge_ts = []
    edge_scaled_abs_decay = []
    edge_scaled_rel_decay = []
    
    for i in range(_true_edges.shape[0]):
        if _true_edges[i]:
            edge_index[0].append(_src[i])
            edge_index[1].append(_dst[i])
            edge_values.append(_link_vals[i])
            edge_ts.append(_ts[i])
            edge_scaled_abs_decay.append(_abs_decay[i])
            edge_scaled_rel_decay.append(_rel_decay[i])
    
    e_index = torch.tensor(edge_index)
    e_values = torch.tensor(edge_values)
    e_ts = torch.tensor(edge_ts)
    e_scaled_abs_decay = torch.tensor(edge_scaled_abs_decay)
    e_scaled_rel_decay = torch.tensor(edge_scaled_rel_decay)
    
    return e_index, e_values, e_ts, e_scaled_abs_decay, e_scaled_rel_decay

def train_test_split_by_user(e_idx, e_vals, e_ts, e_abs_t_decay, e_rel_t_decay, test_size=0.1, seed=0, verbose=False):
    num_users = len(np.unique(e_idx[0]))
    train_e_idx, train_e_vals, train_e_ts, train_e_abs_t_decay, train_e_rel_t_decay, val_e_idx, val_e_vals, val_e_ts, val_e_abs_t_decay, val_e_rel_t_decay = [], [], [], [], [], [], [], [], [], []

    for user in range(num_users):
        # Find interactions for the current user
        user_interactions = np.where(e_idx[0] == user)[0]

        # Split interactions for the current user
        train_indices, val_test_indices = train_test_split(user_interactions, test_size=test_size, random_state=seed)
    
        # Split the edge index and values for the current user
        train_e_idx.append(e_idx[:, train_indices])
        train_e_vals.append(e_vals[train_indices])
        train_e_ts.append(e_ts[train_indices])
        train_e_abs_t_decay.append(e_abs_t_decay[train_indices])
        train_e_rel_t_decay.append(e_rel_t_decay[train_indices])
        val_e_idx.append(e_idx[:, val_test_indices])
        val_e_vals.append(e_vals[val_test_indices])
        val_e_ts.append(e_ts[val_test_indices])
        val_e_abs_t_decay.append(e_abs_t_decay[val_test_indices])
        val_e_rel_t_decay.append(e_rel_t_decay[val_test_indices])
    
    # Concatenate the results for all users
    train_e_idx =  torch.from_numpy(np.concatenate(train_e_idx, axis=1))
    train_e_vals =  torch.from_numpy(np.concatenate(train_e_vals))
    train_e_ts =  torch.from_numpy(np.concatenate(train_e_ts))
    train_e_abs_t_decay =  torch.from_numpy(np.concatenate(train_e_abs_t_decay))
    train_e_rel_t_decay =  torch.from_numpy(np.concatenate(train_e_rel_t_decay))
    val_e_idx =  torch.from_numpy(np.concatenate(val_e_idx, axis=1))
    val_e_vals =  torch.from_numpy(np.concatenate(val_e_vals))
    val_e_ts =  torch.from_numpy(np.concatenate(val_e_ts))
    val_e_abs_t_decay =  torch.from_numpy(np.concatenate(val_e_abs_t_decay))
    val_e_rel_t_decay =  torch.from_numpy(np.concatenate(val_e_rel_t_decay))

    if verbose:
        print("Train size:", len(train_e_vals), "Val size:", len(val_e_vals), "Train ts size:", len(train_e_ts), "Val ts size:", len(val_e_ts))

    return train_e_idx, train_e_vals, train_e_ts, train_e_abs_t_decay, train_e_rel_t_decay, val_e_idx, val_e_vals, val_e_ts, val_e_abs_t_decay, val_e_rel_t_decay


def rmat_2_adjmat(num_users, num_items, edge_index, edge_values, edge_ts, edge_abs_t_decay, edge_rel_t_decay):
    r_M = torch.zeros((num_users, num_items))
    t_M = torch.zeros((num_users, num_items))
    abs_d_M = torch.zeros((num_users, num_items))
    rel_d_M = torch.zeros((num_users, num_items))

    # convert sparse coo format to dense format to get R
    for i in range(len(edge_index[0])):
        row_idx = edge_index[0][i]
        col_idx = edge_index[1][i]
        r_M[row_idx][col_idx] = edge_values[i]  # r_M is the rating matrix
        t_M[row_idx][col_idx] = edge_ts[i]  # t_M is the timestamp matrix
        abs_d_M[row_idx][col_idx] = edge_abs_t_decay[i]  # d_M is the distance matrix
        rel_d_M[row_idx][col_idx] = edge_rel_t_decay[i]  # d_M is the distance matrix

    # perform the r_mat to adj_mat conversion
    r_M_transpose = torch.transpose(r_M, 0, 1)
    t_M_transpose = torch.transpose(t_M, 0, 1)
    abs_d_M_transpose = torch.transpose(abs_d_M, 0, 1)
    rel_d_M_transpose = torch.transpose(rel_d_M, 0, 1)

    r_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    r_adj_mat[:num_users, num_users:] = r_M.clone()
    r_adj_mat[num_users:,: num_users] = r_M_transpose.clone()
    
    t_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    t_adj_mat[:num_users, num_users:] = t_M.clone()
    t_adj_mat[num_users:,: num_users] = t_M_transpose.clone()
    
    abs_d_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    abs_d_adj_mat[:num_users, num_users:] = abs_d_M.clone()
    abs_d_adj_mat[num_users:,: num_users] = abs_d_M_transpose.clone()

    rel_d_adj_mat = torch.zeros((num_users + num_items, num_users + num_items))
    rel_d_adj_mat[:num_users, num_users:] = rel_d_M.clone()
    rel_d_adj_mat[num_users:,: num_users] = rel_d_M_transpose.clone()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    r_adj_mat_coo = r_adj_mat.to_sparse_coo()
    r_adj_mat_coo_indices = r_adj_mat_coo.indices()
    r_adj_mat_coo_values = r_adj_mat_coo.values()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    t_adj_mat_coo = t_adj_mat.to_sparse_coo()
    t_adj_mat_coo_ts = t_adj_mat_coo.values()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    abs_d_adj_mat_coo = abs_d_adj_mat.to_sparse_coo()
    abs_d_adj_mat_coo_decay = abs_d_adj_mat_coo.values()
    
    # convert from dense format back to sparse coo format so we can get the edge_index of adj_mat
    rel_d_adj_mat_coo = rel_d_adj_mat.to_sparse_coo()
    rel_d_adj_mat_coo_decay = rel_d_adj_mat_coo.values()
        
    return r_adj_mat_coo_indices, r_adj_mat_coo_values, t_adj_mat_coo_ts, abs_d_adj_mat_coo_decay, rel_d_adj_mat_coo_decay

def adjmat_2_rmat(num_users, num_items, adj_e_idx, adj_e_vals, adj_e_ts, adj_e_abs_decay, adj_e_rel_decay, verbose = False):
    
    r_sparse_input_edge_index = SparseTensor(row=adj_e_idx[0], 
                                           col=adj_e_idx[1],
                                           value = adj_e_vals,
                                           sparse_sizes=((num_users + num_items), (num_users + num_items))) 
    
    t_sparse_input_edge_index = SparseTensor(row=adj_e_idx[0], 
                                              col=adj_e_idx[1], 
                                              value = adj_e_ts,
                                              sparse_sizes=((num_users + num_items), (num_users + num_items))) 
    
    abs_d_sparse_input_edge_index = SparseTensor(row=adj_e_idx[0], 
                                              col=adj_e_idx[1], 
                                              value = adj_e_abs_decay,
                                              sparse_sizes=((num_users + num_items), (num_users + num_items))) 
    
    rel_d_sparse_input_edge_index = SparseTensor(row=adj_e_idx[0], 
                                              col=adj_e_idx[1], 
                                              value = adj_e_rel_decay,
                                              sparse_sizes=((num_users + num_items), (num_users + num_items))) 
    
    r_adj_mat = r_sparse_input_edge_index.to_dense()
    t_adj_mat = t_sparse_input_edge_index.to_dense()
    abs_d_adj_mat = abs_d_sparse_input_edge_index.to_dense()
    rel_d_adj_mat = rel_d_sparse_input_edge_index.to_dense()
    
    if verbose:
        print("adj_mat: \n", r_adj_mat)
        
    r_interact_mat = r_adj_mat[:num_users, num_users:]
    t_interact_mat = t_adj_mat[:num_users, num_users:]
    abs_d_interact_mat = abs_d_adj_mat[:num_users, num_users:]
    rel_d_interact_mat = rel_d_adj_mat[:num_users, num_users:]

    r_mat_edge_index = r_interact_mat.to_sparse_coo().indices()
    r_mat_edge_values = r_interact_mat.to_sparse_coo().values()
    r_mat_edge_ts = t_interact_mat.to_sparse_coo().values()
    r_mat_edge_abs_decay = abs_d_interact_mat.to_sparse_coo().values()
    r_mat_edge_rel_decay = rel_d_interact_mat.to_sparse_coo().values()
    
    return r_mat_edge_index, r_mat_edge_values, r_mat_edge_ts, r_mat_edge_abs_decay, r_mat_edge_rel_decay