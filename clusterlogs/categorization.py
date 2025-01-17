import numpy as np
import pandas as pd

from functools import reduce
from .utility import levenshtein_similarity_1_to_n


def execute_categorization(df):
    indices = []
    for row in df.itertuples():
        start_index = row.Index
        search_key_phrases_similarities(start_index, df, indices)

    similar_df = pd.DataFrame(indices)
    s = similar_df.copy(deep=True)
    categories = []
    sequence_categorization(s, categories)
    print(categories)

    df["category"] = 0
    for idx, row in enumerate(categories):
        for i in row:
            df.loc[df.index == i, "category"] = idx
    return df.groupby(["category"])


def search_key_phrases_similarities(start, df, indices):
    initial_row = df["common_phrases"][start]
    matched_indices = []
    rows = {}
    for i in initial_row:
        matcher = i
        for row in df.itertuples():
            matching_row = row.common_phrases
            similarities = levenshtein_similarity_1_to_n(matching_row, matcher)
            if similarities == 1.0:
                similarities = [1.0]
            if len([x for x in similarities if x >= 0.9]) > 0:
                matched_indices.append(row.Index)
        print(matched_indices)
        rows[i] = np.unique(matched_indices)
    seq = [v for k, v in rows.items()]
    print(seq)
    merge_arr = []
    if len(seq) != 0:
        merge_arr = list(reduce(set.intersection, [set(v) for k, v in rows.items()]))
    indices.append({"id": start, "indices": list(merge_arr)})


def sequence_categorization(sequences, categories):
    initial_row = sequences.loc[0]
    matcher = np.array(initial_row.indices).tolist()
    if len(matcher) == 0:
        categories.append([sequences.loc[0].id])
        sequences.drop(sequences.index[[0]], inplace=True)
        sequences.reset_index(drop=True, inplace=True)
    else:
        s = [np.array(each).tolist() for each in sequences["indices"].values]
        if len(s) != 0:
            similarities = levenshtein_similarity_1_to_n(s, matcher)
            to_remove = [i for i, x in enumerate(similarities) if x >= 0.9]
            matched_rows = [sequences.loc[i].id for i in to_remove]
            categories.append(matched_rows)
            print("Sequences:")
            print(sequences.shape)
            sequences.drop(sequences.index[to_remove], inplace=True)
            sequences.reset_index(drop=True, inplace=True)

    if sequences.shape[0] > 0:
        sequence_categorization(sequences, categories)
