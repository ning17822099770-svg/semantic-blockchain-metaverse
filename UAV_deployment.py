import collections.abc

import numpy as np
import random
import copy
import A2A_model_get_rate as A2A
import binding_and_A2G_communication_model as bA2G


class UAV_Deployment:
    # parameters

    def __init__(self):
        pass

    def UAV_Running(self, task, group, server, dist, MAPSIZE):
        initial_position = []
        updatad_position = []
        for i in range(len(server)):
            Average_throughput_not_update = self.costfunction_A2A_part(task, server, group, dist)
            initial_position.append([server[i][5][0], server[i][5][1]])
            server[i][5][0], server[i][5][1] = self.PSO(i, task, server, group, dist, MAPSIZE)
            print('initial_position:', initial_position)
            print('Average_throughput_not_update:', Average_throughput_not_update)

            bA = bA2G.Binding_and_A2G()
            distance = bA.calculate_distance(task, server)
            dist = [distance[i][group[i]] for i in range(len(group))]
            Average_throughput_updated = self.costfunction_A2A_part(task, server, group, dist)
            updatad_position.append([server[i][5][0], server[i][5][1]])
            print('updatad_position:', updatad_position)
            print('Average_throughput_updated:', Average_throughput_updated)
        return server, dist

    def CreateRandomSolution(self, VarMin, VarMax):
        sol = {'x': random.uniform(VarMin['x'], VarMax['x']),
               'y': random.uniform(VarMin['y'], VarMax['y'])}
        return sol

    def costfunction_A2G_part(self, task, group, server, dist):
        Average_throughput = np.zeros(len(server))
        # find all users bound to each server
        for i in range(len(server)):
            locals()["list_" + str(i)] = []
            for j in range(len(task)):
                if group[j] == i:
                    current_list = locals()["list_" + str(i)]
                    current_list.append(j)
        # compute the average uplink (collection) and downlink (return) throughput at the current position
        bA = bA2G.Binding_and_A2G()
        A2G_col_rate = bA.A2G_col_rate(task, server, group, dist)
        A2G_ret_rate = bA.A2G_ret_rate(task, server, group, dist)

        for i in range(len(server)):
            current_list = locals()["list_" + str(i)]
            if not current_list:
                Average_throughput[i] = 0
            else:
                for j in current_list:
                    Average_throughput[i] = Average_throughput[i] + 1 / 2 * (A2G_col_rate[j] + A2G_ret_rate[j])
                Average_throughput[i] = Average_throughput[i] / len(current_list)
        return Average_throughput

    def costfunction_A2A_part(self, task, server, group, dist):
        count = np.zeros(len(server))
        Average_throughput_A2A = np.zeros(len(server))
        Average_throughput_A2G = self.costfunction_A2G_part(task, group, server, dist)
        Average_throughput_total = np.zeros(len(server))
        AA = A2A.A2A_model_get_rate()
        A2A_transmit_rate = AA.geneate_A2A_rate(task, server) #new_task

        l = 0
        for j in task:
            h = group[int(j[8][0])]
            # print(h)
            Average_throughput_A2A[h] = Average_throughput_A2A[h] + sum(A2A_transmit_rate[l][h]) / (len(server) - 1)
            count[h] = count[h] + 1
            l = l + 1

        for i in range(len(server)):
            if count[i] != 0:
                Average_throughput_total[i] = 1 / 2 * (Average_throughput_A2A[i] / count[i]) + 1 / 2 * \
                                              Average_throughput_A2G[i]
            else:
                Average_throughput_total[i] = 1 / 2 * Average_throughput_A2G[i]

        return Average_throughput_total.sum()

    def PSO(self, num_server, task, server, group, dist, MAPSIZE):
        MaxIt = 100
        nPop = 50
        # Population Size(Swarm Size)
        w = 1
        # Inertia Weight
        wdamp = 0.98
        # Inertia Weight Damping Ratio
        c1 = 1.5
        # Personal Learning Coefficient
        c2 = 1.5
        # Global Learning Coefficient
        v = 10
        # moving speed of UAV servers
        new_server = server
        BestCost = np.zeros(MaxIt)

        VarMin = {'x': 1, 'y': 1}
        VarMax = {'x': MAPSIZE, 'y': MAPSIZE}

        # Lower and upper Bounds of velocity
        alpha = 0.5
        VelMax_x = alpha * (VarMax['x'] - VarMin['x'])
        VelMin_x = -VelMax_x
        VelMax_y = alpha * (VarMax['y'] - VarMin['y'])
        VelMin_y = -VelMax_y

        # Initialization
        # Create Empty Particle Structure
        empty_particle = {}
        empty_particle['Position'] = []
        empty_particle['Velocity'] = {}
        empty_particle['Velocity']['x'] = 0
        empty_particle['Velocity']['y'] = 0
        empty_particle['Cost'] = -float('inf')
        empty_particle['Best'] = {}
        empty_particle['Best']['Position'] = []
        empty_particle['Best']['Cost'] = -float('inf')
        # TargetInfor.distance, TargetInfor.direction
        # TargetInfor = MoveToTarget(currentState.Position,target);
        # target_x = model.goal[0]
        # target_y = model.goal[1]

        # Initialize Global Best
        GlobalBest = {'Cost': float('-inf')}

        # Create an empty Particles Matrix, each particle is a solution (searching path)
        particle = [copy.deepcopy(empty_particle) for _ in range(nPop)]

        # Initialization Loop
        isInit = False
        while not isInit:
            # print('Initialising...')
            for i in range(nPop):
                # Initialize Position
                particle[i]['Position'] = self.CreateRandomSolution(VarMin, VarMax)

                print('particle[i]position =', particle[i]['Position']['x'])
                print('particle[i]position =', particle[i]['Position']['y'])


                # Initialize Velocity
                particle[i]['Velocity']['x'] = 0
                particle[i]['Velocity']['y'] = 0

                # Evaluation
                new_server[num_server][5][0] = particle[i]['Position']['x']
                new_server[num_server][5][1] = particle[i]['Position']['y']

                bA = bA2G.Binding_and_A2G()
                distance = bA.calculate_distance(task, new_server)
                dist = [distance[i][group[i]] for i in range(len(group))]

                particle[i]['Cost'] = self.costfunction_A2A_part(task, new_server, group, dist)
                print('cost', particle[i]['Cost'])

                # Update Personal Best
                particle[i]['Best']['Position'] = particle[i]['Position']
                particle[i]['Best']['Cost'] = particle[i]['Cost']

                # Update Global Best
                if particle[i]['Best']['Cost'] > GlobalBest['Cost']:
                    GlobalBest = particle[i]['Best']
                    isInit = True

            # PSO loop
            for it in range(MaxIt):
                for i in range(nPop):
                    # x Part
                    # Update Velocity
                    particle[i]['Velocity']['x'] = w * particle[i]['Velocity']['x'] \
                                                   + c1 * np.random.rand() * (
                                                           particle[i]['Best']['Position']['x'] -
                                                           particle[i]['Position']['x']) \
                                                   + c2 * np.random.rand() * (
                                                           GlobalBest['Position']['x'] - particle[i]['Position'][
                                                       'x'])
                    # print('particle_V=', particle[i]['Velocity']['x'])
                    # Update Velocity Bounds
                    particle[i]['Velocity']['x'] = max(particle[i]['Velocity']['x'], VelMin_x)
                    particle[i]['Velocity']['x'] = min(particle[i]['Velocity']['x'], VelMax_x)
                    # print('particle_V=', particle[i]['Velocity']['x'])

                    # Update Position
                    particle[i]['Position']['x'] = particle[i]['Position']['x'] + particle[i]['Velocity']['x']
                    # Velocity Mirroring
                    # If a particle moves out of the range, it will moves backward next time
                    OutOfTheRange = (particle[i]['Position']['x'] < VarMin['x']) or (
                            particle[i]['Position']['x'] > VarMax['x'])
                    # print('OutOfTheRange = ', OutOfTheRange)
                    if isinstance(particle[i]['Velocity']['x'], collections.abc.Sequence):
                        particle[i]['Velocity']['x'][OutOfTheRange] = -particle[i]['Velocity']['x'][OutOfTheRange]
                    else:
                        # Handle the case where particle[i]['Velocity']['x'] is not a sequence
                        # For example, it might be a single float value
                        particle[i]['Velocity']['x'] = -particle[i]['Velocity']['x']

                    # particle[i]['Velocity']['x'][OutOfTheRange] = -particle[i]['Velocity']['x'][OutOfTheRange]
                    # Update Position Bounds
                    particle[i]['Position']['x'] = max(particle[i]['Position']['x'], VarMin['x'])
                    particle[i]['Position']['x'] = min(particle[i]['Position']['x'], VarMax['x'])

                    # y Part
                    # Update Velocity
                    particle[i]['Velocity']['y'] = w * particle[i]['Velocity']['y'] \
                                                   + c1 * np.random.rand() * (
                                                           particle[i]['Best']['Position']['y'] -
                                                           particle[i]['Position']['y']) \
                                                   + c2 * np.random.rand() * (
                                                           GlobalBest['Position']['y'] - particle[i]['Position'][
                                                       'y'])
                    # Update Velocity Bounds
                    particle[i]['Velocity']['y'] = max(particle[i]['Velocity']['y'], VelMin_y)
                    particle[i]['Velocity']['y'] = min(particle[i]['Velocity']['y'], VelMax_y)
                    # Update Position
                    particle[i]['Position']['y'] = particle[i]['Position']['y'] + particle[i]['Velocity']['y']
                    # Velocity Mirroring
                    OutOfTheRange = (particle[i]['Position']['y'] < VarMin['y']) or (
                            particle[i]['Position']['y'] > VarMax['y'])

                    if isinstance(particle[i]['Velocity']['y'], collections.abc.Sequence):
                        particle[i]['Velocity']['y'][OutOfTheRange] = -particle[i]['Velocity']['y'][OutOfTheRange]
                    else:
                        # Handle the case where particle[i]['Velocity']['x'] is not a sequence
                        # For example, it might be a single float value
                        particle[i]['Velocity']['y'] = -particle[i]['Velocity']['y']

                    # Update Position Bounds
                    particle[i]['Position']['y'] = max(particle[i]['Position']['y'], VarMin['y'])
                    particle[i]['Position']['y'] = min(particle[i]['Position']['y'], VarMax['y'])

                    # update the coordinates of the UAV server being optimized by PSO
                    new_server[num_server][5][0] = particle[i]['Position']['x']
                    new_server[num_server][5][1] = particle[i]['Position']['y']

                    # Evaluation
                    bA = bA2G.Binding_and_A2G()
                    distance = bA.calculate_distance(task, new_server)
                    dist = [distance[i][group[i]] for i in range(len(group))]
                    particle[i]['Cost'] = self.costfunction_A2A_part(task, new_server, group, dist)

                    # Update Personal Best
                    if particle[i]['Cost'] > particle[i]['Best']['Cost']:
                        particle[i]['Best']['Position'] = particle[i]['Position']
                        particle[i]['Best']['Cost'] = particle[i]['Cost']

                        # Update Global Best
                        if particle[i]['Best']['Cost'] > GlobalBest['Cost']:
                            GlobalBest = particle[i]['Best']

                # Inertia Weight Damping
                w = w * wdamp

                # Update Best Cost Ever Found
                BestCost[it] = GlobalBest['Cost']

                # Show Iteration Information
                print('Iteration {}: Best Cost = {}'.format(it, BestCost[it]))

        newPoint = [GlobalBest['Position']['x'], GlobalBest['Position']['y']]
        return newPoint[0], newPoint[1]
