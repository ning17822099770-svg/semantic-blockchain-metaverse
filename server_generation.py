import numpy

import random


class Servergenerate:
    def __init__(self):
        pass

    def generate_type(self, k, num):
        """
        Generate a list of length num whose values are drawn from 1..k such that
        every value appears (as close as possible to) the same number of times.
        """
        count = num // k  # number of times each value should appear
        remainder = num % k  # leftover slots that cannot be divided evenly
        elements = list(range(1, k + 1))  # candidate values
        matrix = []  # result list
        for i in range(count):
            # add every value once per round
            for element in elements:
                matrix.append(element)
        # fill the leftover slots
        if remainder > 0:
            random.shuffle(elements)  # shuffle the candidate values
            for i in range(remainder):
                matrix.append(elements[i])
        random.shuffle(matrix)  # shuffle the final order
        return matrix

    def generate_computing_resource(self, low, type_of_server):
        res = []
        for i in range(len(type_of_server)):
            res.append(low - 20 * type_of_server[i])
        return res

    def generate_CPU_parameter(self, low, type_of_server):
        parameter = []
        for i in range(len(type_of_server)):
            parameter.append(low * type_of_server[i])
        return parameter

    def generate_sample_rate(self, up, k, type_of_server):
        sample_rate = []
        for i in range(len(type_of_server)):
            sample_rate.append((up / (k + 1)) * type_of_server[i])
        return sample_rate

    def generate_CPU_frequency(self, up, k, type_of_server):
        CPU_frequency = []
        for i in range(len(type_of_server)):
            CPU_frequency.append((up / (k + 1)) * type_of_server[i])
        return CPU_frequency

    def generate_location(self, low_mapsize, up_mapsize, num_servers):
        l = [[0 for j in range(2)] for i in range(num_servers)]
        for i in range(0, num_servers):
            for j in range(0, 2):
                l[i][j] = random.randint(low_mapsize, up_mapsize)
        return l

    # def generate_random_bandwidth(self, low, up, num_servers):
    #     bandwidth = []
    #     for i in range(num_servers):
    #         n = random.randint(low, up)
    #         bandwidth.append(n)
    #     return bandwidth

    def generate_random_ptr(self, low, up, num_servers):
        ptr = []
        for i in range(num_servers):
            n = random.uniform(low, up)
            ptr.append(n)
        return ptr

    def generate_random_pcol(self, low, up, num_servers):
        pcol = []
        for i in range(num_servers):
            n = random.uniform(low, up)
            pcol.append(n)
        return pcol

    def generate_server(self, type_server, computing_res, CPU_parameter, sample_rate, CPU_frequency, server_location,
                        ptr, pcol, num):
        server = []
        for i in range(num):
            a = [type_server[i], computing_res[i], CPU_parameter[i], sample_rate[i], CPU_frequency[i],
                 server_location[i], ptr[i], pcol[i]]
            server.append(a)
        return server
