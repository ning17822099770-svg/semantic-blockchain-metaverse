import numpy

import random

class Taskgenerate:
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

    def generate_price(self, up, k, type_of_task):
        p = []
        for i in range(len(type_of_task)):
            p.append(up/k * type_of_task[i])
        return p

    def generate_sample_rate(self, up, k, type_of_task):
        sample_rate = []
        for i in range(len(type_of_task)):
            sample_rate.append((up / (k + 1)) * type_of_task[i])
        return sample_rate

    def generate_datasize(self, low_D, up_D, num_task):
        D = []
        for i in range(num_task):
            n = random.randint(low_D, up_D)
            D.append(n)
        return D

    def generate_requirement_T(self, type_of_task):
        req = []
        for i in range(len(type_of_task)):
            if type_of_task[i] == 1:
                req.append(40)
            elif type_of_task[i] == 2:
                req.append(10)
            elif type_of_task[i] == 3:
                req.append(60)
            elif type_of_task[i] == 4:
                req.append(15)
        return req

    def generate_requirement_R(self, type_of_task):
        req = []
        for i in range(len(type_of_task)):
            if type_of_task[i] == 1:
                req.append(1e6)
            elif type_of_task[i] == 2:
                req.append(5e6)
            elif type_of_task[i] == 3:
                req.append(1e7)
            elif type_of_task[i] == 4:
                req.append(2e7)
        return req


    def generate_location(self, low_mapsize, up_mapsize, num_task):
        l = [[0 for j in range(2)] for i in range(num_task)]
        for i in range(0, num_task):
            for j in range(0, 2):
                l[i][j] = random.randint(low_mapsize, up_mapsize)
        return l

    def generate_bandwidth(self, type_of_task):
        bandwidth = []
        for i in range(len(type_of_task)):
            if type_of_task[i] == 1:
                bandwidth.append(5e6)
            elif type_of_task[i] == 2:
                bandwidth.append(2.5e7)
            elif type_of_task[i] == 3:
                bandwidth.append(5e7)
            elif type_of_task[i] == 4:
                bandwidth.append(1e8)
        return bandwidth
    #  1.5e7, 2e7, 1.5e8, 3e8

    def generate_task(self, type, req_T, req_R, p, D, l, B, sample_rate, num):
        task = []
        for i in range(num):
            a = [type[i], req_T[i], req_R[i], p[i], D[i], l[i], B[i], sample_rate[i], [i]]
            task.append(a)
        return task
