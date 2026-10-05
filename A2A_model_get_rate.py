import numpy as np
import math
import random


class A2A_model_get_rate:
    def __init__(self):
        pass

    def geneate_A2A_rate(self, task, server):

        distance = [[0 for i in range(len(server))] for j in range(len(server))]
        x_server = [row[5][0] for row in server]
        y_server = [row[5][1] for row in server]
        for j in range(len(x_server)):
            for i in range(len(x_server)):
                distance[j][i] = math.sqrt((x_server[j] - x_server[i]) ** 2 + (y_server[j] - y_server[i]) ** 2)

        collecting_channel_param = {'suburban': (4.88, 0.43, 0.1, 21),
                                    'urban': (9.61, 0.16, 1, 20),
                                    'dense-urban': (12.08, 0.11, 1.6, 23),
                                    'high-rise-urban': (27.23, 0.08, 2.3, 34)}

        collecting_params = collecting_channel_param['urban']
        a = collecting_params[0]
        b = collecting_params[1]
        yita0 = collecting_params[2]
        yita1 = collecting_params[3]
        carrier_f = 2.5e9
        noise_power = 1e-13


        fspl = 0
        L = 0
        rate = [[[0 for i in range(len(server))] for h in range(len(server))] for j in range(len(task))]

        for j in range(len(task)):
            # W.append(1e6 * task[i][6])
            for h in range(len(server)):
                for i in range(len(server)):
                    fspl = ((4 * np.pi * carrier_f * distance[h][i] / 3e8) ** 2)
                    L = (fspl * 10 ** (yita0 / 20))
                    if L != 0:
                        rate[j][h][i] = (task[j][6] * np.log2(1 + server[h][6] / (L * noise_power * task[j][6])))
        return rate
