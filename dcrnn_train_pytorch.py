from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import matplotlib.pyplot as plt

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f, Loader=yaml.CLoader)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

        supervisor.train()
        if args.plot_sensor != -1:
          mean_loss, mean_rmse, mean_mape, results = supervisor.evaluate()
          prediction = results["prediction"]
          actual = results["truth"]
          sensor = args.plot_sensor
          plt.figure()
          plt.plot(range(len(prediction[0][:, sensor])), prediction[0][:, sensor], label="Prediction")
          plt.plot(range(len(actual[0][:, sensor])), actual[0][:, sensor], label="Actual")
          plt.legend()
          plt.savefig(f'figures/sensor_{args.plot_sensor}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--plot_sensor', default=-1, type=int, help='Sensor you want to plot.')

    args = parser.parse_args()
    main(args)
