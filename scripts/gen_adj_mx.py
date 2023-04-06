from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import numpy as np
import pandas as pd
import pickle
import random


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1, num_sensors_used=100):
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """

    sensor_ids = sensor_ids[:num_sensors_used]        # Reduce number of sensors used


    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx


def get_adjacency_matrix_tokyo(distance_df, normalized_k=0.1, num_sensors_used=100):
    N = len(distance_df)
    # select random sensors
    indices = random.sample(range(N), k=num_sensors_used)

    # sample distance matrix by selected sensors
    dist_mx = np.zeros((num_sensors_used, num_sensors_used), dtype=np.float32)
    for r in range(num_sensors_used):
        for c in range(num_sensors_used):
            dist_mx[r][c] = distance_df[indices[r]][indices[c]]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str, default='data/sensor_graph/graph_sensor_ids.txt',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='data/sensor_graph/distances_la_2012.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--normalized_k', type=float, default=0.1,
                        help='Entries that become lower than normalized_k after normalization are set to zero for sparsity.')
    parser.add_argument('--output_pkl_filename', type=str, default='data/sensor_graph/adj_mat.pkl',
                        help='Path of the output file.')
    
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    
    # tokyo flag argument
    parser.add_argument('--tokyo', default=False, type=bool,
                        help='True if using TOKYO dataset')
    
    args = parser.parse_args()

    with open(args.sensor_ids_filename) as f:
        sensor_ids = f.read().strip().split(',')
    if args.tokyo:
        #../tokyo/expy-tky_adjdis.npy
        distance_df = np.load(args.distances_filename)
    else:
        distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})


    f = open(args.config_filename)
    supervisor_config = yaml.load(f, Loader=yaml.CLoader)
    num_sensors_used = supervisor_config['model'].get('num_nodes')

    # check if tokyo
    if args.tokyo:
        sensor_ids = []
        sensor_id_to_ind = []
        adj_mx = get_adjacency_matrix_tokyo(distance_df, num_sensors_used=num_sensors_used)
    else:
        _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids, num_sensors_used=num_sensors_used)


    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
