from tqdm import tqdm
import os
import numpy as np
import torch
import pandas as pd
from itertools import product
from tqdm import tqdm
import torch
import utils
import json


def mr1(preds, target):
    """
    Compute the mean rank for relevant items in the predictions.
    Args:
        preds (torch.Tensor): A tensor of predicted scores (higher scores indicate more relevant items).
        target (torch.Tensor): A tensor of true relationships (0 for irrelevant, 1 for relevant).
    Returns:
        float: The mean rank of relevant items for each query.
    """
    has_positives = torch.sum(target, 1) > 0
    
    _, spred = torch.topk(preds, preds.size(1), dim=1)
    found = torch.gather(target, 1, spred)
    temp = torch.arange(preds.size(1)).float() * 1e-6
    _, sel = torch.topk(found - temp, 1, dim=1)
    
    sel = sel.float()
    sel[~has_positives] = torch.nan
    
    return torch.nanmean((sel+1).float())


def csi_relationship_matrix(df):
    
    # Get the number of rows in the DataFrame
    N = len(df)

    # Initialize the matrix with default 'misc'
    matrix = np.full((N, N), 'misc', dtype='<U10')  # Adjust the string length as needed

    # Indices where set_id is the same
    same_set_id_indices = (df['set_id'].values[:, None] == df['set_id'].values[None, :])

    # Indices where nlabel > 1 and seed is True
    nlabel_seed_indices = (df['nlabel'].values[:, None] > 1) & (df['seed'].values[None, :])

    # Set 'shs-pos' where both conditions are met
    matrix[same_set_id_indices & nlabel_seed_indices] = 'shs-pos'

    # Set 'yt-pos' where nlabel > 1 and seed conditions are partially met
    yt_pos_indices = same_set_id_indices & (df['nlabel'].values[:, None] > 1) & (df['nlabel'].values[None, :] > 1) & (~nlabel_seed_indices)
    matrix[yt_pos_indices] = 'yt-pos'

    # Set 'yt-neg' 
    yt_neg_indices = (df['nlabel'].values[:, None] > 1) & (df['nlabel'].values[None, :] == 1)
    matrix[same_set_id_indices & yt_neg_indices] = 'yt-neg'

    # set 'yt-nomusic'
    yt_nomusic_indices = (df['nlabel'].values[:, None] == 0) | (df['nlabel'].values[None, :] == 0)
    matrix[yt_nomusic_indices] = 'yt-nomusic'
    
    # Set 'random-neg' where set_id is different and both labels are > 1
    shs_neg_indices = (~same_set_id_indices) & (df['nlabel'].values[:, None] > 1) & (df['nlabel'].values[None, :] > 1)
    matrix[shs_neg_indices] = 'random-neg'

    # any remaining relationships where one is other but both have different set_ids
    matrix[(matrix == 'misc') & (df['nlabel'].values[:, None] > 0) & (df['nlabel'].values[:, None] > 0)] = 'random-neg'

    # Set diagonal to 'self'
    np.fill_diagonal(matrix, 'self')

    # Make the matrix symmetric
    matrix = np.where(matrix != 'misc', matrix, matrix.T)

    return matrix


def get_ytrue_by_rels(df):
    """Gets the ordinal ytrue tensor for given dataset.
    Args:
        df (_type_): _description_
    Returns:
        _type_: _description_
    """
    rel_matrix = csi_relationship_matrix(df)
    return torch.tensor(np.where(
        np.isin(rel_matrix, ['shs-pos', 'yt-pos']), 
        2, 
        np.where(
            np.isin(rel_matrix, ['self', 'yt-nomusic']), 
            0, 
            1
            )))


def get_dataset_subset(model, dataset):

    data_preds, ytrue, ypred = get_dataset(model, "SHS-SEED+YT")

    data = pd.read_csv(os.path.join("data", f'{dataset}.csv'), sep=';')

    mask = data_preds.yt_id.isin(data.yt_id).values

    ytrue = ytrue[mask, :]
    ytrue = ytrue[:, mask]

    ypred = ypred[mask, :]
    ypred = ypred[:, mask]

    return data, ytrue, ypred


def get_dataset(model, dataset):
    
    if dataset == "SHS-SEED+YT":
        
        return get_dataset_shs_seed_yt(model, False)
    
    else:
        
        base_path = 'data/preds'
        data_path = os.path.join(base_path, model, dataset)

        # metadata file
        data = pd.read_csv(os.path.join("data", f'{dataset}.csv'), sep=';')

        # model predictions
        ypred = torch.load(os.path.join(data_path, 'ypred.pt'))
        ytrue = torch.load(os.path.join(data_path, 'ytrue.pt'))

        # duplicate removal 
        not_duplicated_mask = ~data.yt_id.duplicated(keep='first')
        data = data[not_duplicated_mask]
            
        ypred = ypred[not_duplicated_mask][:, not_duplicated_mask]
        ytrue = ytrue[not_duplicated_mask][:, not_duplicated_mask]
        
        return data, ytrue, ypred


def get_dataset_dfs(model, dataset):
	
    
    df, ytrue, ypred = get_dataset(model, dataset)
    preds_df = pd.DataFrame(ypred, index=df.yt_id, columns=df.yt_id)
    
    df_annot = pd.read_csv("data/SHS-YT.csv", sep=";")
    df = pd.merge(df, df_annot[["set_id", "yt_id", "category_expert", "music_ratio"]], on=["set_id", "yt_id"], how="left")

    return df, preds_df
    

def get_pair_df(model, dataset):
    
    _, preds_df = get_dataset_dfs(model, dataset)
    
    df_melt = pd.melt(preds_df, value_vars=preds_df.columns, ignore_index=False).rename(
    {"yt_id": "yt_id_a"}, axis=1).reset_index().rename(
    {"yt_id": "yt_id_b"}, axis=1)[["yt_id_a", "yt_id_b", "value"]]
    
    return df_melt
    
def get_all_pair_dfs(dataset):
    
    df, preds_df_ch = get_dataset_dfs("coverhunter", dataset)
    df.loc[(df.seed == True) & (df.sample_group.isna() & (df.label == "Version")), "label"] = "SHS-Version"
    df.loc[~(df.sample_group.isna()) & (df.label == "Version"), "label"] = "YT-Version"

    df_ch = get_pair_df("coverhunter", dataset)
    df_cq = get_pair_df("cqtnet", dataset)
    df_rm = get_pair_df("remove", dataset)
    df_di = get_pair_df("ditto", dataset)
    df_fz = get_pair_df("ditto", dataset)

    df_all = pd.merge(
	    df_ch[["yt_id_a", "yt_id_b", "value"]].rename({"value": "cos_ch"}, axis=1),
	    df_cq[["yt_id_a", "yt_id_b", "value"]].rename({"value": "cos_cq"}, axis=1),
	    on=["yt_id_a", "yt_id_b"],
	    how="left")
    
    df_all = pd.merge(
	    df_all,
	    df_rm[["yt_id_a", "yt_id_b", "value"]].rename({"value": "cos_rm"}, axis=1),
	    on=["yt_id_a", "yt_id_b"],
	    how="left")
	
    df_all = pd.merge(
	    df_all,
	    df_di[["yt_id_a", "yt_id_b", "value"]].rename({"value": "cos_di"}, axis=1),
	    on=["yt_id_a", "yt_id_b"],
	    how="left")
    
    df_all = pd.merge(
	    df_all,
	    df_fz[["yt_id_a", "yt_id_b", "value"]].rename({"value": "cos_fz"}, axis=1),
	    on=["yt_id_a", "yt_id_b"],
	    how="left")

    df_all = pd.merge(
    pd.merge(
        df_all, 
        df[["yt_id", "set_id", "seed", "sample_group", "label", "category_expert"]].add_suffix("_a"),
        how="left",
        on="yt_id_a"),
    df[["yt_id", "set_id", "seed", "sample_group", "label", "category_expert"]].add_suffix("_b"),
    how="left",
    on="yt_id_b")

    return df_all.drop_duplicates(subset=["yt_id_a", "yt_id_b"])
    

def get_agg_sim(model):
    """This function gets the mean similarity of candidates to their seed items. 
    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """    
    data, ytrue, ypred = get_dataset(model, "SHS-SEED+YT")
    
    csi_rel_matrix = csi_relationship_matrix(data)

    mask = torch.tensor((csi_rel_matrix == 'yt-neg') | 
                        (csi_rel_matrix == 'yt-pos') | 
                        (csi_rel_matrix == 'shs-pos') | 
                        (csi_rel_matrix == 'yt-nomusic'))

    ypred = torch.where(mask, ypred, torch.nan)
    
    col_name = "_sim_"
    if model == 'coverhunter':
        col_name += "ch"
    elif model == 'cqtnet':
        col_name += "cqt"
    elif model == 'remove':
        col_name += "rmv"
    max_col = 'max' + col_name
    mean_col = 'mean' + col_name
    min_col = 'min' + col_name
    
    data[mean_col] = torch.nanmean(ypred, dim=0)
    data[max_col] =  torch.max(torch.nan_to_num(ypred, 0), dim=0).values
    data[min_col] =  torch.min(torch.nan_to_num(ypred, 0), dim=0).values
    return data[["set_id", "yt_id", mean_col, max_col, min_col]]


def get_shs_yt_agg_sim(models=["coverhunter", "cqtnet", "remove"]):
    
    data = pd.read_csv("data/SHS-YT.csv", sep=";")

    for model in models:
        
        data_model = get_agg_sim(model)
        data = pd.merge(data, data_model, on=["set_id", "yt_id"], how="left")

    return data
 
def get_dataset_seedq(model, dataset):
    
    data, ytrue, ypred = get_dataset(model, dataset)
    csi_rels = csi_relationship_matrix(data)

    # limit on x-axis
    
    ytrue = ytrue[data.seed.values]
    ypred = ypred[data.seed.values]
    csi_rels = csi_rels[data.seed.values]
    data = data.query("seed")
    
    return data, csi_rels, ytrue, ypred


def get_dataset_shs_seed_yt(model, seedq=False):
    
    dataset = 'SHS-SEED+YT'
    data_path = 'data'
    preds_path = os.path.join(data_path, 'preds', model, dataset)

    # metadata file
    data = pd.read_csv(os.path.join(data_path, f"{dataset}.csv"), sep=';').query("has_cqt_ch & has_cqt_20 & has_crema")

    # model predictions
    ypred = torch.load(os.path.join(preds_path, 'ypred.pt'))
    ytrue = get_ytrue_by_rels(data)
    
    # duplicate removal 
    not_duplicated_mask = (~data.duplicated(subset=["set_id", "yt_id"]))
    not_in_shs100k_train_val = (~data.in_shs100k_train & ~data.in_shs100k_train)
    filter_mask = not_duplicated_mask & not_in_shs100k_train_val
    
    data = data[filter_mask]
        
    ypred = ypred[filter_mask.values][:, filter_mask.values]
    ytrue = ytrue[filter_mask.values][:, filter_mask.values]
    
    if seedq:
        ytrue = ytrue[data.seed.values]
        ypred = ypred[data.seed.values]
        data = data[data.seed]
    return data, ytrue, ypred    


def get_rels_ytrue(df, rel):
    
    rel_matrix = csi_relationship_matrix(df)
    mask = rel_matrix == rel
    return torch.tensor(mask.astype(int))


def argsort_rowwise(matrix_a, matrix_b):
    
    matrix_a = np.array(matrix_a)
    matrix_b = np.array(matrix_b)

    num_rows = matrix_a.shape[0]
    sorted_matrix_b = np.empty_like(matrix_b)

    for i in range(num_rows):
        # Calculate the indices to sort the i-th row of matrix_a descendingly
        sorted_indices = np.argsort(matrix_a[i])[::-1]
        
        # Use the sorted_indices to reorder the i-th row of matrix_b
        sorted_matrix_b[i] = matrix_b[i, sorted_indices]

    return sorted_matrix_b


# rank-analysis
def get_rels_df(data, ypred, rels, to_csv=True):
    rels = rels[data.seed]
    ypred = ypred[data.seed.values]

    rels_sorted = utils.argsort_rowwise(ypred, rels)
    df = pd.DataFrame(rels_sorted, index=data.query("seed")[
        ["set_id", "seed", "yt_id", "label"]])
    if to_csv:
        df.to_csv("data/ranks.csv", sep=";")
    return df


def get_rank_cls(df_rels):
    
    df = pd.DataFrame()
    
    for col in df_rels:
        df = pd.concat([df, df_rels[col].value_counts()], axis=1)
                  
    return df  


def first_occurrence_position(row, value):
    index = (row == value).idxmax()
    return None if np.isnan(index) else int(index) + 1


def get_first_ranks_cls(rels_df):
    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate through each row in your original DataFrame
    for index, row in rels_df.iterrows():
        # Create a dictionary to store the results for the current row
        row_result = {}
        
        # Iterate through unique values in the row
        for value in row.unique():
            # Calculate the position of the first occurrence for each value
            position = first_occurrence_position(row, value)
            
            # Add the result to the row_result dictionary
            row_result[f'Position_{value}'] = position
        
        # Convert the row_result dictionary to a DataFrame and append it to the result_df
        row_result_df = pd.DataFrame([row_result])
        result_df = pd.concat([result_df, row_result_df], ignore_index=True)
    
    return result_df
