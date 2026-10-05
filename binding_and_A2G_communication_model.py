import numpy as np
import math
import random


class Binding_and_A2G:
    def __init__(self):
        pass

    # def generate_task_server_group(self, task, server):
    #     """
    #     Build a j-by-i distance matrix.
    #
    #     Find the minimum of each row of the j-by-i matrix and return its column index.
    #
    #     Args:
    #         task: list of tasks
    #         server: list of servers
    #
    #     Returns:
    #         A list of length j holding the column index of each row's minimum.
    #     """
    #     distance = [[0 for i in range(len(server))] for j in range(len(task))]
    #     x_task = [row[5][0] for row in task]
    #     y_task = [row[5][1] for row in task]
    #     x_server = [row[5][0] for row in server]
    #     y_server = [row[5][1] for row in server]
    #     for j in range(len(x_task)):
    #         for i in range(len(x_server)):
    #             distance[j][i] = math.sqrt((x_task[j] - x_server[i]) ** 2 + (y_task[j] - y_server[i]) ** 2)
    #
    #     task_server_group = []
    #     for row in range(len(distance)):
    #         min_val = float('inf')
    #         min_index = -1
    #         for col in range(len(distance[row])):
    #             if distance[row][col] < min_val:
    #                 min_val = distance[row][col]
    #                 min_index = col
    #         task_server_group.append(min_index)
    #
    #     return task_server_group, distance
    def calculate_distance(self, task, server):
        """
        Compute the distance between every task (user) and every server.

        Entry [j][i] of the returned 2-D array is the Euclidean distance between task j and server i.

        Args:
            task: list of tasks
            server: list of servers

        Returns:
            A 2-D array with the Euclidean distance between each task and each server.
        """
        distance = [[0 for i in range(len(server))] for j in range(len(task))]
        x_task = [row[5][0] for row in task]
        y_task = [row[5][1] for row in task]
        x_server = [row[5][0] for row in server]
        y_server = [row[5][1] for row in server]
        for j in range(len(x_task)):
            for i in range(len(x_server)):
                distance[j][i] = math.sqrt((x_task[j] - x_server[i]) ** 2 + (y_task[j] - y_server[i]) ** 2)
                if distance[j][i] == 0:
                    distance[j][i] = 0.00000000001
        return distance

    def get_task_server_group(self, distance):
        """
        Bind each task to its closest server.

        For each row of the distance array, find the column index of the minimum value.

        Args:
            distance: 2-D array with the distance between each task and each server.

        Returns:
            A list holding, for each task, the index of the closest server.
        """
        task_server_group = []
        for row in range(len(distance)):
            min_val = float('inf')
            min_index = -1
            for col in range(len(distance[row])):
                if distance[row][col] < min_val:
                    min_val = distance[row][col]
                    min_index = col
            task_server_group.append(min_index)
        return task_server_group

    def A2G_col_rate(self, task, server, group, dist):  # to be completed

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
        h = 50
        noise_power = 1e-13

        W = []
        Pl = []
        fspl = []
        rate = []
        L = []

        for i in range(len(task)):
            W.append(task[i][6])

            Pl.append(1 / (1 + a * np.exp(-b * (np.arctan(h / dist[i]) - a))))

            fspl.append((4 * np.pi * carrier_f * dist[i] / (3e8)) ** 2)

            L.append(Pl[i] * fspl[i] * 10 ** (yita0 / 20) + 10 ** (yita1 / 20) * fspl[i] * (1 - Pl[i]))

            rate.append(W[i] * np.log2(1 + server[group[i]][7] / (L[i] * noise_power * W[i])))

        return rate

    def A2G_ret_rate(self, task, server, group, dist):  # to be completed

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
        h = 50
        noise_power = 1e-13

        W = []
        Pl = []
        fspl = []
        rate = []
        L = []

        for i in range(len(task)):
            W.append(task[i][6])

            Pl.append(1 / (1 + a * np.exp(-b * (np.arctan(h / dist[i]) - a))))

            fspl.append((4 * np.pi * carrier_f * dist[i] / (3e8)) ** 2)

            L.append(Pl[i] * fspl[i] * 10 ** (yita0 / 20) + 10 ** (yita1 / 20) * fspl[i] * (1 - Pl[i]))

            rate.append(W[i] * np.log2(1 + server[group[i]][6] / (L[i] * noise_power * W[i])))

        return rate

    def sort_with_index(self, lst):
        sorted_lst = sorted(enumerate(lst), key=lambda x: x[1])
        sorted_vals = [tup[1] for tup in sorted_lst]
        sorted_indices = [tup[0] for tup in sorted_lst]
        return sorted_vals, sorted_indices

    def reorder(self, task, order):
        sorted_task = [tup[0] for tup in sorted(zip(task, order), key=lambda x: x[1])]
        return sorted_task


