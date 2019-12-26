import pandas as pd
import sys
import pprint
from clusterlogs import pipeline, cluster_output

def main():
    df = pd.read_csv('error_messages.csv', index_col=0)
    df.set_index('pandaid', inplace=True)
    # To specify clustering parameters, please use dictionary:
    # clustering_parameters = {'tokenizer':'nltk',
    #                          'w2v_size': 300,
    #                          'w2v_window': 10,
    #                          'min_samples': 1}
    target = 'exeerrordiag'
    mode = 'INDEX'
    cluster = pipeline.ml_clustering(df, target, mode='create', model_name='word2vec_test.model')
    cluster.process()

    # output
    output = cluster_output.Output(df, target)
    output.clustered_output(mode)
    stats = output.statistics(output_mode='frame')
    pprint.pprint(stats)
    #
    pprint.pprint(stats['pattern'].tolist())

    pprint.pprint(output.postprocessing())
    # output = cluster.clustered_output(mode)
    # stats = cluster.statistics(output_mode='dict')

    # pprint.pprint(cluster.cluster_labels)
    # pprint.pprint(output)
    # pprint.pprint(stats)
    #
    # pprint.pprint(cluster.in_cluster(1))
    #
    # pprint.pprint(cluster.messages_cleaned)
    # pprint.pprint(cluster.tokenized)
    # pprint.pprint(cluster.epsilon)
    # pprint.pprint(cluster.timings)
    # cluster.distance_curve(cluster.distances, 'save')

if __name__ == "__main__":
    sys.exit(main())