import editdistance
import difflib
import numpy as np
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer


STATISTICS = ["cluster_name",
              "cluster_size",
              "pattern",
              "sequence",
              #"mean_length",
              "mean_similarity",
              #"std_length",
              "std_similarity"]

class Output:

    def __init__(self, df, target, level=1):
        self.df = df
        self.target = target
        self.level = level
        self.stat_df = None
        self.stat_dict = None
        self.hierarchy = None



    def clustered_output(self, type='idx', level=1):
        """
        Returns dictionary of clusters with the arrays of elements
        :return:
        """
        groups = {}
        for key, value in self.df.groupby(['cluster_'+str(level)]):
            if type == 'all':
                # groups[str(key)] = value.to_dict(orient='records')
                groups[str(key)] = value
            elif type == 'idx':
                groups[str(key)] = value.index.values.tolist()
            elif type == 'target':
                groups[str(key)] = value[self.target].values.tolist()
            elif type == 'cleaned':
                groups[str(key)] = value['cleaned'].values.tolist()
        return groups


    # def in_cluster(self, cluster_label, level=1):
    #     """
    #     Returns all log messages in particular cluster
    #     :param cluster_label:
    #     :return:
    #     """
    #     return self.df[self.df['cluster_'+str(level)] == cluster_label][self.target].values


    def levenshtein_similarity(self, rows):
        """
        Takes a list of log messages and calculates similarity between
        first and all other messages.
        :param rows:
        :return:
        """
        return ([100 - (editdistance.eval(rows[0], rows[i]) * 100) / len(rows[0]) for i in range(0, len(rows))])


    def statistics(self, output_mode='frame', level=1, restruct=True):
        """
        Returns dictionary with statistic for all clusters
        "cluster_name" - name of a cluster
        "cluster_size" = number of log messages in cluster
        "pattern" - all common substrings in messages in the cluster
        "mean_length" - average length of log messages in cluster
        "std_length" - standard deviation of length of log messages in cluster
        "mean_similarity" - average similarity of log messages in cluster
        "std_similarity" - standard deviation of similarity of log messages in cluster
        "interbals" - internal indexes of the cluster
        :param clustered_df:
        :param output_mode: frame | dict
        :return:
        """
        clusters = []
        clustered_df = self.clustered_output('all', level)
        for item in clustered_df:
            cluster = clustered_df[item]
            patterns = []
            self.tokenized_patterns(item, cluster, patterns, level, restruct)
            clusters.extend(patterns)
        return pd.DataFrame(clusters, columns=STATISTICS)\
            .round(2)\
            .sort_values(by='cluster_size', ascending=False)



    def positioning(self, tokens):
        return [(v, k) for k, v in enumerate(tokens)]


    def tokenized_patterns(self, item, cluster, results, level=1, restruct=True):

        if cluster.shape[0] == 1:
            value = {'cluster_name': item,
                     'cluster_size': 1,
                     'pattern': cluster.iloc[0]['cleaned'],
                     'sequence': cluster.iloc[0]['tokenized_treebank'],
                     'mean_similarity': 1,
                     'std_similarity': 0}
            results.append(value)
            return results
        else:
            curr = cluster.iloc[0]['tokenized_treebank']
            # curr = cluster.iloc[0][self.target]
            outliers, commons, similarity = [],[],[]
            [self.matcher(curr, row, outliers, commons, similarity, restruct) for row in cluster.itertuples()]
            # [self.matcher_lines(curr, row, outliers, commons, similarity, restruct) for row in cluster.itertuples()]
            # self.rematch(commons)
            commons = [i[0] for i in sorted(set([val for sublist in commons for val in sublist]), key=lambda x: x[1])]
            # pattern = self.rematch(commons)
            twd = TreebankWordDetokenizer()

            value = {'cluster_name': item,
                     'cluster_size': cluster.shape[0] - len(outliers),
                     #'pattern': pattern,
                     'pattern': twd.detokenize(commons),
                     'sequence': commons,
                     'mean_similarity': np.mean(similarity),
                     'std_similarity': np.std(similarity)}

            results.append(value)

            if len(outliers) == 0:
                return results
            else:
                new_item = int(self.df['cluster_'+str(level)].max()) + 1
                self.df.loc[outliers, 'cluster_'+str(level)] = new_item
                self.tokenized_patterns(new_item, self.df.loc[outliers], results, level)


    #TODO
    def rematch(self, commons):
        """
        First we found matches in strings.
        Next we find matches in matches.
        :param commons:
        :return:
        """
        common_sequence = []
        lines = []
        twd = TreebankWordDetokenizer()
        arr = {}
        sorted_arr = [sorted(set([val for sublist in commons for val in sublist]), key=lambda x: x[1])]
        for i in sorted_arr:
            for k,v in i:
                if arr.get(v):
                    arr[v].append(k)
                else:
                    arr[v] = []
                    arr[v].append(k)
        print(arr)
        # common_sequence.append(self.multimatcher(arr[pos]))
        # lines.append(twd.detokenize(common_sequence))



    def multimatcher(self, sequence):
        common = []
        for i in sequence:
            for j in sequence:
                matches = difflib.SequenceMatcher(None, i, j)
                match = matches.find_longest_match(
                    0, len(i), 0, len(j))
                common.append(i[match.a:match.size])
        min_length = np.min([len(line) for line in common])
        return [len(line) == min_length for line in common][0]


    def matcher(self, current, row, outliers, commons, similarity, restruct=True):

        matches = difflib.SequenceMatcher(None, current, row.tokenized_treebank)
        if restruct == True:
            if matches.ratio() <= 0.5:  # this line does not belong to this cluster - move to new cluster
                twd = TreebankWordDetokenizer()
                print('Outliers detected with ratio {} for messages \n {} \n and \n {}'.format(matches.ratio(), twd.detokenize(current), twd.detokenize(row.tokenized_treebank)))
                outliers.append(row.Index)
            else:
                similarity.append(matches.ratio())
                common = [current[m.a:m.a + m.size] for m
                          in matches.get_matching_blocks() if m.size > 0]
                flat = [val for sublist in common for val in sublist]
                commons.append(self.positioning(flat))
        else:
            similarity.append(matches.ratio())
            common = [current[m.a:m.a + m.size] for m
                      in matches.get_matching_blocks() if m.size > 0]
            flat = [val for sublist in common for val in sublist]
            commons.append(self.positioning(flat))


    def matcher_lines(self, current, row, outliers, commons, similarity, restruct=True):

        matches = difflib.SequenceMatcher(None, current, row.cleaned)
        if restruct == True:
            if matches.ratio() <= 0.5:  # this line does not belong to this cluster - move to new cluster
                print('Outliers detected with ratio {} for messages \n {} \n and \n {}'.format(matches.ratio(), current, row.cleaned))
                outliers.append(row.Index)
            else:
                similarity.append(matches.ratio())
                common = [current[m.a:m.a + m.size] for m
                          in matches.get_matching_blocks() if m.size > 0]
                #flat = [val for sublist in common for val in sublist]
                commons.append(self.positioning(common))
        else:
            similarity.append(matches.ratio())
            common = [current[m.a:m.a + m.size] for m
                      in matches.get_matching_blocks() if m.size > 0]
            #flat = [val for sublist in common for val in sublist]
            commons.append(self.positioning(common))


    def reclustering(self, df, result):
        """

        :param df:
        :param updated_clusters:
        :return:
        """
        # select the first pattern
        sequences = df['sequence'].values
        matches = [difflib.SequenceMatcher(None, sequences[0], x) for x in sequences]
        df['ratio'] = [item.ratio() for item in matches]
        filtered = df[(df['ratio'] >= 0.5)]['cluster_name'].values
        result.append(filtered)
        df.drop(df[df['cluster_name'].isin(filtered)].index, inplace=True)
        while df.shape[0] > 0:
            self.reclustering(df, result)


    def postprocessing(self, stat_df, level=1):
        """
        Clustering the results of the first clusterization
        :return:
        """
        # sort statistics df by cluster size in ascending order
        sorted_df = stat_df.sort_values(by=['cluster_size'])[['cluster_size',
                                                                   'cluster_name',
                                                                   'pattern',
                                                                   'sequence']]
        result = []
        self.reclustering(sorted_df, result)
        new_level = []
        for k,v in enumerate(result):
            x = self.df[self.df['cluster_'+str(level)].isin(v)].index
            new_level.append({'cluster_name': k, 'idx': x})
        for cluster in new_level:
            self.df.loc[cluster['idx'], 'cluster_'+str(level+1)] = str(cluster['cluster_name'])

