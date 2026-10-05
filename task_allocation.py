import numpy as np
import pandas as pd
import random
from random import choice


# random.seed(1)


class Task_allocation:
    # parameters
    # task is a matrix which contains [price p_j, data size D_j, executing time t_e]
    # server is a list, which contains the computational powers for each server
    def __init__(self, server, R, task, group, m, n, ALPHA, GAMMA, EPSILON,
                 MAX_EPISODES, lamda, rho, nu, deata, omiga, account):
        self.task = task
        self.server = server
        self.server_type = [server[i][0] for i in range(len(server))]
        self.lamda = lamda
        self.rho = rho
        self.nu = nu
        self.deata = deata
        self.omiga = omiga
        self.account = account
        self.group = group
        self.m = m  # number of tasks
        self.n = n  # number of servers
        # self.B = [server[i][8] for i in range(len(server))] # bandwidth
        # self.P = [server[i][6] for i in range(len(server))]  # trainsimission power
        self.alpha = [server[i][2] for i in range(len(server))]  # CPU parameter
        self.f = [server[i][4] for i in range(len(server))]  # CPU frequency
        self.sample = [server[i][3] for i in range(len(server))]  # sample cpu cycles
        self.R = R
        self.N_STATES = self.m  # number of states
        self.ACTIONS = [i for i in range(self.n)]  # action
        self.EPSILON = EPSILON  # greedy
        self.ALPHA = ALPHA  # learning rate
        self.GAMMA = GAMMA  # discount
        self.MAX_EPISODES = MAX_EPISODES
        self.q_table = self.build_q_table()  # q_table
        self.mismatches = np.zeros((1, len(server)))

    # create the q_table with initial value=0
    def build_q_table(self):
        q_table = pd.DataFrame(
            np.zeros((self.N_STATES, len(self.ACTIONS))),  # initialize q_table
            columns=self.ACTIONS,  # columns, the name of actions
        )
        return q_table

    # pandas.DataFrame creates a 2-D table with an ordered set of columns; each column may hold a different value type

    # check whether the server i can process task j with cpu, throughput and time constraints
    def check_avaliable(self, j, i, h):
        if self.task[j][4] >= self.server[i][1]:  # not enough CPU capacity
            return 0

        if self.server[i][0] != self.task[j][0]:
            if self.task[j][4] > self.deata * self.server[i][1]:  # mismatched-type workload exceeds the delta * capacity quota
                return 0

        if self.R[j][h][i] < self.task[j][2]:  # throughput requirement not met
            return 0

        if (self.task[j][4] * self.sample[i] / self.f[i] + self.task[j][4] * 1e6 / self.R[j][h][i]).all() > \
                self.task[j][1]:
            return 0

        return 1

    # if j is from i, then no transimision
    def check_avaliable_1(self, j, i):
        if self.task[j][4] >= self.server[i][1]:  # not enough CPU capacity
            return 0

        if self.server[i][0] != self.task[j][0]:
            if self.task[j][4] > self.deata * self.server[i][1]:  # mismatched-type workload exceeds the delta * capacity quota
                return 0

        if self.task[j][4] / self.f[i] > self.task[j][1]:
            return 0

        return 1

    # find the source server that collected task j
    def find_source(self, j):
        if j < len(self.group):
            return self.group[j]
        else:
            return None

    # check whether task j is from server i
    def check_source(self, j, i):
        if self.group[j] == i:
            return 1
        else:
            return 0

    # get the reward based on sources and availability (implements the utility functions)
    def get_feedback(self):
        reward = 0
        rew = []
        for j in range(self.m):  # for each task
            for i in range(self.n):  # for each server
                if self.check_source(j, i) == 1:  # task was collected by this server
                    # h=i
                    if self.check_avaliable_1(j, i) == 1:  # task can be computed locally
                        if self.task[j][0] == self.server[i][0]:
                            reward = self.task[j][3] * self.task[j][4] * self.sample[i] - self.server[i][2] * \
                                     self.task[j][4] * self.sample[i] * self.server[i][4] ** 2
                        elif self.task[j][0] != self.server[i][0]:
                            reward = self.task[j][3] * self.task[j][4] * self.sample[i] - self.server[i][2] * \
                                     self.task[j][4] * self.sample[i] * self.server[i][4] ** 2 - self.rho * \
                                     self.task[j][3] * self.task[j][4] * self.sample[i]
                    elif self.check_avaliable_1(j, i) != 1:
                        reward = 0  # placeholder, updated later when used
                if self.check_source(j, i) != 1:  # task was collected by another server
                    h = self.find_source(j)  # find the task's source server
                    if self.check_avaliable(j, i, h) == 1:  # forwarding is feasible, assign reward by case
                        if self.task[j][0] == self.server[i][0]:
                            reward = (1 - self.lamda) * self.task[j][3] * self.task[j][4] * self.sample[i] - self.server[i][2] * \
                                     self.task[j][4] * self.sample[i] * self.server[i][4] ** 2 - self.server[i][6] * \
                                     self.task[j][4] * 1e6 / self.R[j][i][h]
                        if self.task[j][0] != self.server[i][0]:
                            reward = (1 - self.lamda) * self.task[j][3] * self.task[j][4] * self.sample[i] - self.server[i][2] *\
                                     self.task[j][4] * self.sample[i] * self.server[i][4] ** 2 - self.server[i][6] *\
                                     self.task[j][4] * 1e6 / self.R[j][i][h] - self.rho * self.task[j][3] * self.task[j][
                                        4] * self.sample[i]
                    else:
                        reward = 0  # get nothing because it can't process task j
                rew.append(reward)  # using a list to contain all the rewards for each task and each server (forwarding-source reward not yet updated)
        return rew

    # The rewards above exclude the communication penalty (nu); it is added after an action is selected

    # get the set of available actions for each task: candidate servers with a positive reward
    def check_action_reward(self, task_num):
        reward = self.get_feedback()
        reward = np.array(reward).reshape(self.m, self.n)  # using a m*n matrix to contain the rewards
        index_list = []
        for i in range(self.n):  # for each server
            if reward[task_num][i] > 0:  # if =0, the server can't process that task
                action_index = i
                index_list.append(action_index)
        return index_list  # return the avaliable index of actions !=0first

        # to get the available actions set for each task without speed up: treat every server as a candidate

    # epsilon-greedy: pick the action with the largest Q value for the task
    def select_action(self, task_num, state_actions):
        # state_actions = self.check_action_reward(task_num)
        if (np.random.uniform() > self.EPSILON) or (len(state_actions) == 0):
            # if np.random.uniform() > self.EPSILON:  # not greedy
            # action_name = np.random.choice(state_actions)
            action_name = choice(state_actions)

        else:
            q_value = pd.DataFrame(self.q_table.loc[task_num, state_actions]).T
            q_value = q_value.fillna(-1)
            action_name = int(q_value.idxmax(axis=1))  # greedy

        return action_name

    # create a table to contain the cpu capacity
    def cpu_table(self):
        cpu_table = self.build_q_table()
        return cpu_table

    # create a table to contain the time
    def time_table(self):
        time_table = self.build_q_table()
        return time_table

    def mismatch_table(self):
        mismatch_table = self.build_q_table()
        return mismatch_table

    # Check whether the selected server is valid, i.e. whether it can still take this task given the tasks it has
    # already been assigned. If the computing constraints are violated, remove the server from the candidate list
    # and re-select the server with the largest Q value, until a valid one is found; record it in actions_ava.

    def check_action_valid(self, j, actions_ava, rewards, cpu_table, mismatch_table, state_actions, action, acc_mu,
                           acc_mismatch):
        if action in actions_ava:  # select actions in available sets
            if (self.task[j][0] == self.server[action][0] and acc_mu + self.task[j][4] > self.server[action][1]) or \
                    (self.task[j][0] != self.server[action][0] and (
                            acc_mu + self.task[j][4] > self.server[action][1] or acc_mismatch + self.task[j][
                        4] > self.deata * self.server[action][1])):
                if action in state_actions:
                    state_actions.remove(action)  # remove that action from state_actions
                if not state_actions:  # if state_actions is empty
                    return None
                action = self.select_action(j, state_actions)

                if (self.task[j][0] == self.server[action][0] and acc_mu + self.task[j][4] > self.server[action][1]) or \
                        (self.task[j][0] != self.server[action][0] and (
                                acc_mu + self.task[j][4] > self.server[action][1] or acc_mismatch + self.task[j][
                            4] > self.deata * self.server[action][1])):
                    return self.check_action_valid(j, actions_ava, rewards, cpu_table, mismatch_table,
                                                   state_actions,
                                                   action, acc_mu, acc_mismatch)
        else:
            actions_ava.append(action)
            return self.check_action_valid(j, actions_ava, rewards, cpu_table, mismatch_table,
                                           state_actions,
                                           action, acc_mu, acc_mismatch)
        return action

    # def check_action_valid(self, j, actions_ava, rewards, cpu_table, mismatch_table, state_actions, action, acc_mu,
    #                        acc_mismatch):
    #     res = True
    #     if action in actions_ava:  # select actions in avaliable sets
    #         if (self.task[j][0] == self.server[action][0] and acc_mu + self.task[j][4] > self.server[action][1]) or (
    #                 self.task[j][0] != self.server[action][0] and (
    #                 acc_mu + self.task[j][4] > self.server[action][1] or acc_mismatch + self.task[j][4] > self.deata *
    #                 self.server[action][1])):
    #             state_actions.remove(action)  # remove that action
    #             action = self.select_action(j, state_actions)
    #             if (self.task[j][0] == self.server[action][0] and acc_mu + self.task[j][4] > self.server[action][
    #                 1]) or (self.task[j][0] != self.server[action][0] and (
    #                     acc_mu + self.task[j][4] > self.server[action][1] or acc_mismatch + self.task[j][
    #                 4] > self.deata * self.server[action][1])):
    #                 action = self.check_action_valid(j, actions_ava, rewards, cpu_table, mismatch_table, state_actions,
    #                                                  action, acc_mu, acc_mismatch)
    #         else:
    #             actions_ava.append(action)
    #     if not state_actions:  # if state_actions is empty
    #         return None
    #     return action

    def q_update_modified(self):
        count = 0
        bad_count = 0
        actions_ava = []
        rews = []
        rew1s = []
        act = []
        tasks_not_serverd = []
        rewards = np.array(self.get_feedback()).reshape(self.m, self.n)
        cpu_table = self.cpu_table()
        mismatch_table = self.mismatch_table()

        for j in range(self.m):
            h = self.find_source(j)
            state_actions = self.check_action_reward(j)  # candidate servers with a positive reward
            if state_actions:
                action = self.select_action(j, state_actions)  # server with the largest Q value
            else:
                action = None

            if action is not None:

                # initialize acc_t and acc_mu
                acc_mu = cpu_table[action].sum()  # sum of cpu
                acc_mismatch = mismatch_table[action].sum()

                action = self.check_action_valid(j, actions_ava, rewards, cpu_table, mismatch_table, state_actions, action,
                                                 acc_mu, acc_mismatch)

            if action is None:
                action = h
                tasks_not_serverd.append(j)
                rewards[j][h] = 0
                bad_count = bad_count + 1

            # record the cpu and time
            cpu_table[action][j] = self.task[j][4]
            acc_mu = cpu_table[action].sum()  # sum of cpu

            if self.task[j][0] != self.server[action][0]:
                mismatch_table[action][j] = self.task[j][4]
                acc_mismatch = mismatch_table[action].sum()
            if self.check_source(j, action) != 1:
                if self.task[j][0] == self.server[action][0]:
                    rewards[j][h] = self.lamda * self.task[j][3] * self.task[j][4] * self.sample[action] - \
                                    self.server[h][6] * self.task[j][4] * 1e6 / self.R[j][h][action]
                elif self.task[j][0] != self.server[action][0]:
                    rewards[j][h] = self.lamda * self.task[j][3] * self.task[j][4] * self.sample[action] - \
                            self.server[h][6] * self.task[j][4] * 1e6 / self.R[j][h][action] - self.nu * self.task[j][3] * \
                            self.task[j][4] * self.sample[action]
            # update q
            if self.check_source(j, action) != 1:
                h = self.find_source(j)
                if j != self.m - 1:
                    self.q_table.iloc[j, action] += self.ALPHA * (rewards[j][action] + self.account * rewards[j][h]
                                                                  + self.GAMMA * np.max(
                                self.q_table.iloc[j + 1, state_actions]) - self.q_table.iloc[j, action])
                else:
                    self.q_table.iloc[j, action] += self.ALPHA * (rewards[j][action] + self.account * rewards[j][h]
                                                                  + self.GAMMA * np.max(
                                self.q_table.iloc[j, state_actions]) - self.q_table.iloc[j, action])

                self.q_table.iloc[j, h] = rewards[j][h]

            else:
                if j != self.m - 1:
                    self.q_table.iloc[j, action] += self.ALPHA * (rewards[j][action]
                                                                  + self.GAMMA * np.max(
                                self.q_table.iloc[j + 1, state_actions]) - self.q_table.iloc[j, action])
                else:
                    self.q_table.iloc[j, action] += self.ALPHA * (rewards[j][action]
                                                                  + self.GAMMA * np.max(
                                self.q_table.iloc[j, state_actions]) - self.q_table.iloc[j, action])
            rew = rewards[j][action]  # the rewards of each step
            rews.append(rew)  # rewads for all the tasks
            act.append(action)  # action set

            if self.check_source(j, action) != 1:
                h = self.find_source(j)
                rew1 = rewards[j][h]
                rew1s.append(rew1)
            elif self.check_source(j, action) == 1:
                rew1s.append(0)

            if self.task[j][0] == self.server[action][0] and j not in tasks_not_serverd:
                count = count + 1

        res1 = sum(rew1s)
        matchingrate = count / self.m
        serverd_rate = (self.m - bad_count) / self.m
        res = sum(rews)  # total rewards for one tempt
        return self.q_table, rews, res, res1, act, rew1s, tasks_not_serverd, matchingrate, serverd_rate

    # training with speed up
    def training(self):
        res = []
        reward_compute = []
        reward_forward = []
        act = []
        matching = []
        no_serve = []
        serving_rate = []
        Q = []

        # forwarding = []
        count = 0
        # training
        for i in range(self.MAX_EPISODES):
            q_table, rews, reward_c, reward_f, actions, forwarding_reward, tasks_not_serverd, matchingrate, serverd_rate = self.q_update_modified()
            Q.append(q_table)
            no_serve.append(tasks_not_serverd)
            matching.append(matchingrate)
            serving_rate.append(serverd_rate)
            res.append(reward_c + reward_f)
            act.append(actions)
            reward_compute.append(rews)
            reward_forward.append(forwarding_reward)

        max_reward = np.max(res)
        best_solution = act[res.index(np.max(res))]
        matchrate = matching[res.index(np.max(res))]
        not_serving = no_serve[res.index(np.max(res))]
        serverate = serving_rate[res.index(np.max(res))]
        Q_table = Q[res.index(np.max(res))]
        reward_computing = reward_compute[res.index(np.max(res))]
        reward_fordwarding = reward_forward[res.index(np.max(res))]

        index_list = [j for j in range(self.m)]

        task_type = [self.task[j][0] for j in index_list]
        server_type = [self.server[j][0] for j in best_solution]

        best_server_task = list(
            zip(index_list, self.group, best_solution, task_type, server_type))  # the best allocation

        return max_reward, best_server_task, matchrate, Q_table, res, reward_computing, reward_fordwarding, not_serving, serverate

    # Greedy baseline: among servers that satisfy the computing constraints, choose the one with the largest reward
    def check_action_valid_greedy(self, j, h, actions_ava, rewards, cpu_table, mismatch_table, state_actions, action,
                                  acc_mu, acc_mismatch):
        new_rewards = []
        if action in actions_ava:  # select actions in available sets
            if (self.task[j][0] == self.server[action][0] and acc_mu + self.task[j][4] > self.server[action][1]) or \
                    (self.task[j][0] != self.server[action][0] and (
                            acc_mu + self.task[j][4] > self.server[action][1] or acc_mismatch + self.task[j][
                        4] > self.deata * self.server[action][1])):
                if action in state_actions:
                    state_actions.remove(action)  # remove that action from state_actions
                if not state_actions:  # if state_actions is empty
                    return None
                for i in state_actions:
                    new_rewards.append(rewards[j][i] + rewards[j][h])
                max_reward_index = np.argmax(new_rewards)
                action = state_actions[max_reward_index]
                if (self.task[j][0] == self.server[action][0] and acc_mu + self.task[j][4] > self.server[action][1]) or \
                        (self.task[j][0] != self.server[action][0] and (
                                acc_mu + self.task[j][4] > self.server[action][1] or acc_mismatch + self.task[j][
                            4] > self.deata * self.server[action][1])):
                    return self.check_action_valid_greedy(j, h, actions_ava, rewards, cpu_table, mismatch_table,
                                                          state_actions,
                                                          action, acc_mu, acc_mismatch)
        else:
            actions_ava.append(action)
            return self.check_action_valid_greedy(j, h, actions_ava, rewards, cpu_table, mismatch_table,
                                                  state_actions,
                                                  action, acc_mu, acc_mismatch)
        return action

    # get the index based on greedy
    def greedy_reward(self, j, h, rewards):
        # rewards = pd.DataFrame(rewards)
        index = self.check_action_reward(j)
        print(index)
        if index:
            reward = np.zeros(self.n)
            source_reward = np.zeros(self.n)
            # reward = rewards.iloc[j, index]
            # new_reward = reward.copy()
            for i in index:
                if i != h:
                    if self.task[j][0] == self.server[i][0]:
                        source_reward[i] = self.lamda * self.task[j][3] * self.task[j][4] * self.sample[i] - \
                                self.server[h][6] * self.task[j][4] * 1e6 / self.R[j][h][i]
                    elif self.task[j][0] != self.server[i][0]:
                        source_reward[i] = self.lamda * self.task[j][3] * self.task[j][4] * self.sample[i] - \
                                self.server[h][6] * self.task[j][4] * 1e6 / self.R[j][h][i] - self.nu * self.task[j][
                                    3] * self.task[j][4] * self.sample[i]
                    reward[i] = source_reward[i] + rewards[j][i]
                elif i == h:
                    reward[i] = rewards[j][i]
            max_reward = max(list(reward))
            if max_reward <= 0:
                return None
            else:
                max_index = list(reward).index(max_reward)
                if max_index != h:
                    rewards[j][h] = source_reward[max_index]
                return max_index
        else:
            return None

    # get the rewards by greedy search
    def greedy_select(self):
        count = 0
        bad_count = 0
        actions_ava = []
        rews = []
        rew1s = []
        act = []
        tasks_not_serverd = []
        rewards = np.array(self.get_feedback()).reshape(self.m, self.n)
        cpu_table = self.cpu_table()
        mismatch_table = self.mismatch_table()
        # check the limitation of time and cpu
        for j in range(self.m):
            h = self.find_source(j)
            state_actions = self.check_action_reward(j)
            action = self.greedy_reward(j, h, rewards)

            if action is not None:
                # initialize acc_t and acc_mu
                acc_mu = cpu_table[action].sum()  # sum of cpu
                acc_mismatch = mismatch_table[action].sum()

                action = self.check_action_valid_greedy(j, h, actions_ava, rewards, cpu_table, mismatch_table,
                                                        state_actions,
                                                        action, acc_mu, acc_mismatch)

            if action is None:
                action = h
                tasks_not_serverd.append(j)
                rewards[j][h] = 0
                bad_count = bad_count + 1

            # record the cpu and time: update the computing resources of the selected server
            cpu_table[action][j] = self.task[j][4]
            acc_mu = cpu_table[action].sum()  # sum of cpu

            if self.task[j][0] != self.server[action][0]:
                mismatch_table[action][j] = self.task[j][4]
                acc_mismatch = mismatch_table[action].sum()

            act.append(action)
            if self.task[j][0] == self.server[action][0] and j not in tasks_not_serverd:
                count = count + 1

            res = rewards[j][action]
            rews.append(res)

            if self.check_source(j, action) != 1:
                rew1 = rewards[j][h]
                rew1s.append(rew1)
            elif self.check_source(j, action) == 1:
                rew1s.append(0)

        serverd_rate = (self.m - bad_count) / self.m
        matchrate = count / self.m
        rews_sum = sum(rews)
        rews1_sum = sum(rew1s)
        res = rews_sum + rews1_sum

        index_list = [j for j in range(self.m)]
        task_type = [self.task[j][0] for j in index_list]
        server_type = [self.server[j][0] for j in act]

        best_server_task = list(
            zip(index_list, self.group, act, task_type, server_type))  # the best allocation

        return res, best_server_task, matchrate, rews, rew1s, tasks_not_serverd, serverd_rate

    # Random baseline: choose any server that satisfies the computing constraints
    def check_action_valid_random(self, j, actions_ava, rewards, cpu_table, mismatch_table, state_actions, action,
                                  acc_mu,
                                  acc_mismatch):
        if action in actions_ava:  # select actions in available sets
            if (self.task[j][0] == self.server[action][0] and acc_mu + self.task[j][4] > self.server[action][1]) or \
                    (self.task[j][0] != self.server[action][0] and (
                            acc_mu + self.task[j][4] > self.server[action][1] or acc_mismatch + self.task[j][
                        4] > self.deata * self.server[action][1])):
                if action in state_actions:
                    state_actions.remove(action)  # remove that action from state_actions
                if not state_actions:  # if state_actions is empty
                    return None

                action = choice(state_actions)
                if (self.task[j][0] == self.server[action][0] and acc_mu + self.task[j][4] > self.server[action][1]) or \
                        (self.task[j][0] != self.server[action][0] and (
                                acc_mu + self.task[j][4] > self.server[action][1] or acc_mismatch + self.task[j][
                            4] > self.deata * self.server[action][1])):
                    return self.check_action_valid_random(j, actions_ava, rewards, cpu_table, mismatch_table,
                                                          state_actions,
                                                          action, acc_mu, acc_mismatch)
        else:
            actions_ava.append(action)
            return self.check_action_valid_random(j, actions_ava, rewards, cpu_table, mismatch_table,
                                                  state_actions,
                                                  action, acc_mu, acc_mismatch)
        return action

    # get the index randomly selected
    def random_index(self, j):
        # rewards = pd.DataFrame(rewards)
        # print('rewards', rewards)
        # for j in range(self.m):
        index = self.check_action_reward(j)
        # reward = rewards.iloc[j, index]
        # print('reward', reward)
        if index:
            random_index = random.choice(index)
            # random_reward = choice(list(reward))
            # random_index = list(reward).index(random_reward)
            return random_index
        else:
            return None

    # get the rewards by random strategy
    def random_select(self):
        count = 0
        bad_count = 0
        actions_ava = []
        rews = []
        rew1s = []
        act = []
        tasks_not_serverd = []

        rewards = np.array(self.get_feedback()).reshape(self.m, self.n)
        cpu_table = self.cpu_table()
        mismatch_table = self.mismatch_table()
        # check the limitation of time and cpu
        for j in range(self.m):
            h = self.find_source(j)
            state_actions = self.check_action_reward(j)
            action = self.random_index(j)

            if action is not None:
                # initialize acc_t and acc_mu
                acc_mu = cpu_table[action].sum()  # sum of cpu
                acc_mismatch = mismatch_table[action].sum()

                action = self.check_action_valid_random(j, actions_ava, rewards, cpu_table, mismatch_table,
                                                        state_actions,
                                                        action,
                                                        acc_mu, acc_mismatch)

            if action is None:
                action = h
                tasks_not_serverd.append(j)
                rewards[j][h] = 0
                bad_count = bad_count + 1

            # record the cpu and time
            cpu_table[action][j] = self.task[j][4]
            acc_mu = cpu_table[action].sum()  # sum of cpu

            if self.task[j][0] != self.server[action][0]:
                mismatch_table[action][j] = self.task[j][4]
                acc_mismatch = mismatch_table[action].sum()

            if self.check_source(j, action) != 1:
                if self.task[j][0] == self.server[action][0]:
                    rewards[j][h] = self.lamda * self.task[j][3] * self.task[j][4] * self.sample[action] - \
                            self.server[h][6] * self.task[j][4] * 1e6 / self.R[j][h][action]
                elif self.task[j][0] != self.server[action][0]:
                    rewards[j][h] = self.lamda * self.task[j][3] * self.task[j][4] * self.sample[action] - \
                            self.server[h][6] * self.task[j][4] * 1e6 / self.R[j][h][action] - self.nu * \
                            self.task[j][3] * self.task[j][4] * self.sample[action]
            # update the forwarding reward
            res = rewards[j][action]
            rews.append(res)

            if self.check_source(j, action) != 1:
                h = self.find_source(j)
                rew1 = rewards[j][h]
                rew1s.append(rew1)
            elif self.check_source(j, action) == 1:
                rew1s.append(0)

            act.append(action)
            if self.task[j][0] == self.server[action][0] and j not in tasks_not_serverd:
                count = count + 1

        matchrate = count / self.m
        rews_sum = sum(rews)
        rews1_sum = sum(rew1s)
        res = rews_sum + rews1_sum

        serverd_rate = (self.m - bad_count) / self.m

        index_list = [j for j in range(self.m)]
        task_type = [self.task[j][0] for j in index_list]
        server_type = [self.server[j][0] for j in act]

        best_server_task = list(
            zip(index_list, self.group, act, task_type, server_type))  # the best allocation
        return res, best_server_task, matchrate, rews, rew1s, tasks_not_serverd, serverd_rate
