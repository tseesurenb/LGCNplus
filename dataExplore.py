import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dataPrep as dp
import random

# Generate timestamps
timestamps = [random.randint(0, 2500) for i in range(2500)]

# Create DataFrame
rating_df = pd.DataFrame({'timestamp': timestamps})
rating_df["timestamp"] = rating_df["timestamp"].astype("int64")
_start = rating_df['timestamp'].min()
_end = rating_df['timestamp'].max()

print(f'min:{_start}, max:{_end}')

_total_dist = _end - _start 

_dist_unit = 1 # one day
 # hyperparameter that defines time distance weight
exp_beta = 0.01

_bias = 0 #0.0001
    
_beta = 0.05
rating_df['u_abs_decay_linear'] = _bias + np.power(((rating_df['timestamp'] - _start) / _dist_unit), 1) # linear
rating_df['u_abs_decay_log'] = _bias + np.power(((rating_df['timestamp'] - _start) / _dist_unit), _beta) # log
rating_df['u_abs_decay_recip'] = _bias + np.power(((rating_df['timestamp'] - _start) / _dist_unit), 1/_beta) # reciprocal
rating_df['u_abs_decay_exp'] = _bias + np.exp(-exp_beta * (rating_df['timestamp'] - _start) / _dist_unit) # exp

print(rating_df)

sorted_values = sorted(rating_df['u_abs_decay_linear'])
#plt.plot(sorted_values, label = 'linear')
    
sorted_values = sorted(rating_df['u_abs_decay_log'])
#plt.plot(sorted_values, label = 'log')

sorted_values = sorted(rating_df['u_abs_decay_recip'])
#plt.plot(sorted_values, label = 'recip')

sorted_values = sorted(rating_df['u_abs_decay_exp'])
plt.plot(sorted_values, label = 'exp')

# Add labels and title
plt.xlabel('Index (after sorting)')
plt.ylabel('u_abs_decay')
plt.title('Sorted Plot of u_abs_decay')
# Add legend
plt.legend(loc='upper left')

# Set the y-axis limits to be between 1 and 1.1
#plt.ylim(0, 5e-44)
#plt.xlim(0, 5)


# Show the plot
plt.show()


# ------------------------- Relative -------------------------- #
'''
local_agg_emb_len = 0
new_df = None
_win_unit = 24*3600 # 1 day
_beta = g_r_beta

    
# Convert timestamp to int64
rating_df["timestamp"] = rating_df["timestamp"].astype("int64")

# Find the minimum and maximum timestamp for each user
user_min_timestamp = rating_df.groupby('userId')['timestamp'].min()
user_max_timestamp = rating_df.groupby('userId')['timestamp'].max()

new_df_list = []
local_agg_emb_len_list = []

# Iterate over each user
for user_id, (user_start_date, user_end_date) in zip(user_min_timestamp.index, zip(user_min_timestamp, user_max_timestamp)):
    # Calculate _dist for each user
    _dist = float((user_end_date - user_start_date) / _win_unit)
    
    # Calculate new timestamp for each user
    rating_df.loc[rating_df['userId'] == user_id, 'u_rel_decay'] = (1 + ((rating_df['timestamp'] - user_start_date) / _win_unit)) ** (_beta)
    
    # Create a new DataFrame for the user
    new_user_df = rating_df.loc[rating_df['userId'] == user_id, ['userId', 'itemId', 'rating', 'timestamp', 'u_abs_decay_koren', 'u_abs_decay_exp', 'u_abs_decay_power','u_rel_decay']]
    
    # Append the DataFrame to the list
    new_df_list.append(new_user_df)
    
    # Calculate the length of unique new_u_ts_id values for each user
    local_agg_emb_len_list.append(len(new_user_df['u_rel_decay'].unique()))

# Concatenate DataFrames for all users
new_df = pd.concat(new_df_list, ignore_index=True)

# Get the sorted values of 'new_u_ts_id'
#sorted_values = sorted(new_df['u_rel_decay'])
#plt.plot(sorted_values, label = 'relative decay')

# Add labels and title
plt.xlabel('Index (after sorting)')
plt.ylabel('u_rel_decay')
plt.title('Sorted Plot of u_rel_decay')
plt.legend()

# Show the plot
plt.show()
'''