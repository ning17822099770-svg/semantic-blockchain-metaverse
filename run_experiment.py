"""
Run the penalty-sweep experiments of the paper (Section V, Fig. 3 and Fig. 4).

For every random seed this script
  1. generates the semantic tasks and the UAV servers,
  2. binds every user to its closest UAV server,
  3. optionally optimizes the UAV positions with PSO (problem P1),
  4. computes the A2G / A2A channel rates,
  5. sweeps the computing penalty rho (Fig. 3) or the communication penalty nu (Fig. 4)
     and solves the task allocation (problem P2) with Q-learning, greedy and random selection,
  6. stores one CSV per (seed, penalty value) under test_values/.

Examples:
    python run_experiment.py --sweep rho                 # Q-learning + PSO, sweep rho
    python run_experiment.py --sweep nu --no-uav-opt     # without UAV position optimization, sweep nu
    python run_experiment.py --sweep rho --seeds 0 1 2
"""
import argparse
import os
import random

import numpy as np
import pandas as pd

import task_generation as tg
import UAV_deployment as Ud
import task_allocation as ta
import server_generation as sg
import A2A_model_get_rate as A2A
import binding_and_A2G_communication_model as bA2G

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Penalty values swept in the paper
SWEEP_VALUES = {
    'rho': np.arange(0, 1, 0.05),  # computing penalty coefficient
    'nu': np.arange(0, 0.1, 0.01),  # communication penalty coefficient
}


def Initialization(seed_number, sweep, uav_opt, values, out_dir):
    # Set random seed
    random.seed(seed_number)

    # Set constants
    k = 4
    num_task = 60
    num_servers = 20
    map_size = 200

    # Generate tasks with different semantics
    data = tg.Taskgenerate()
    type_task = data.generate_type(k, num_task)
    p = data.generate_price(20, k, type_task)
    D = data.generate_datasize(20, 30, num_task)
    req_T = data.generate_requirement_T(type_task)
    req_R = data.generate_requirement_R(type_task)
    l = data.generate_location(0, map_size, num_task)
    B = data.generate_bandwidth(type_task)
    sample_rate = data.generate_sample_rate(10, k, type_task)
    task = data.generate_task(type_task, req_T, req_R, p, D, l, B, sample_rate, num_task)

    # Generate UAV servers with different computing abilities
    se = sg.Servergenerate()
    type_server = se.generate_type(k, num_servers)
    computing_res = se.generate_computing_resource(160, type_server)
    CPU_parameter = se.generate_CPU_parameter(0.01, type_server)
    sample_rate = se.generate_sample_rate(10, k, type_server)
    CPU_frequency = se.generate_CPU_frequency(20, k, type_server)
    server_location = se.generate_location(0, map_size, num_servers)
    ptr = se.generate_random_ptr(30, 50, num_servers)
    pcol = se.generate_random_pcol(5, 10, num_servers)

    server = se.generate_server(type_server, computing_res, CPU_parameter, sample_rate, CPU_frequency, server_location,
                                ptr,
                                pcol, num_servers)

    print('server_old =', server)
    # Bind the initial binding of UAV and user based on distance, calculate the A2G channel rate, collection and return time, and reorder task upload order based on collection time.
    bA = bA2G.Binding_and_A2G()
    distance = bA.calculate_distance(task, server)
    group = bA.get_task_server_group(distance)

    dist = [distance[i][group[i]] for i in range(len(group))]

    # Update the UAV positions with PSO after binding (problem P1)
    if uav_opt:
        UD = Ud.UAV_Deployment()
        server, dist = UD.UAV_Running(task, group, server, dist, map_size)
        print('server_new =', server)

    A2G_col_rate = bA.A2G_col_rate(task, server, group, dist)
    A2G_ret_rate = bA.A2G_ret_rate(task, server, group, dist)

    col_time = [1e6 * task[i][4] / A2G_col_rate[i] for i in range(len(task))]
    ret_time = [1e6 * task[i][4] / A2G_ret_rate[i] for i in range(len(task))]

    new_col_time, order = bA.sort_with_index(col_time)
    new_task = bA.reorder(task, order)
    new_ret_time = bA.reorder(ret_time, order)
    new_group = bA.reorder(group, order)

    # Build the A2A model and compute the rates between servers. Use new_task, since it is the input to the
    # task allocation below.
    AA = A2A.A2A_model_get_rate()
    A2A_transmit_rate = AA.geneate_A2A_rate(new_task, server)

    for value in values:
        rho = value if sweep == 'rho' else 0
        nu = value if sweep == 'nu' else 0
        run_allocation(rho, nu, sweep, uav_opt, out_dir, server, A2A_transmit_rate, new_task, new_group, num_task,
                       num_servers, seed_number)


def run_allocation(rho, nu, sweep, uav_opt, out_dir, server, A2A_transmit_rate, new_task, new_group, num_task,
                   num_servers, seed_number):
    # Start training
    EPSILON = 0.9  # greedy
    GAMMA = 0.9  # discount
    ALPHA = 0.5  # learning rate
    max_iteration = 500  # iterations 500
    lamda = 0.1
    deata = 0.3
    omiga = 0.6
    account = 1
    price = 5
    resource = 160_140_120_100

    # NOTE: Task_allocation's signature is (..., ALPHA, GAMMA, EPSILON, ...). The arguments are passed here as
    # (EPSILON, ALPHA, GAMMA), exactly as in the code that produced the results in test_values/, so the
    # effective values inside Task_allocation are ALPHA=0.9, GAMMA=0.5, EPSILON=0.9.
    ql = ta.Task_allocation(server, A2A_transmit_rate, new_task, new_group, num_task, num_servers, EPSILON, ALPHA,
                            GAMMA,
                            max_iteration, lamda, rho, nu, deata, omiga, account)

    # Greedy select
    greedy_max_reward, greedy_solution, greedy_matchrate, greedy_compute_reward, greedy_forward_reward, greedy_tasks_not_serverd, greedy_serverd_rate = ql.greedy_select()
    print('Greedy max reward:', greedy_max_reward)
    print('Greedy match rate:', greedy_matchrate)
    print('Greedy tasks_not_serverd:', greedy_tasks_not_serverd)
    print('Greedy tasks_serverd_rate:', greedy_serverd_rate)
    print('Greedy solution:', greedy_solution)
    print('Greedy_compute_reward:', greedy_compute_reward)
    print('Greedy_forward_reward:', greedy_forward_reward)

    # Random select
    random_max_reward, random_solution, random_matchrate, random_compute_reward, random_forward_reward, random_tasks_not_serverd, random_serverd_rate = ql.random_select()
    print('Random max reward:', random_max_reward)
    print('Random match rate:', random_matchrate)
    print('Random tasks_not_serverd:', random_tasks_not_serverd)
    print('Random tasks_serverd_rate:', random_serverd_rate)
    print('Random solution:', random_solution)
    print('Random_compute_reward :', random_compute_reward)
    print('Random_fordward_reward:', random_forward_reward)

    # Q-Learning
    Q_max_reward, Q_solution, Q_matchrate, Q_table, Q_res, Q_compute_reward, Q_fordward_reward, Q_not_serving, Q_serverate = ql.training()

    print('Q-learning match rate:', Q_matchrate)
    print('Q-learning max reward:', Q_max_reward)
    print('Q-learning task_serving_rate:', Q_serverate)
    print('Q-learning not serving_task:', Q_not_serving)
    print('Q-learning solution:', Q_solution)
    print('Q_compute_reward :', Q_compute_reward)
    print('Q_fordward_reward:', Q_fordward_reward)

    # Save the results
    df = pd.DataFrame({
        'seed_number': [seed_number],
        'num_task': [num_task],
        'num_server': [num_servers],
        'lamda': [lamda],
        'rho': [rho],
        'nu': [nu],
        'deata': [deata],
        'omiga': [omiga],
        'ALPHA': [ALPHA],
        'GAMMA': [GAMMA],
        'EPSILON': [EPSILON],
        'MAX_EPISODES': [max_iteration],
        'Q_max_reward': [Q_max_reward],
        'Q_solution': [Q_solution],
        'Q_matchrate': [Q_matchrate],
        'q_table': [Q_table],
        'Q_serverate': [Q_serverate],
        'Q_not_serving': [Q_not_serving],
        'Q_compute_reward': [Q_compute_reward],
        'Q_fordward_reward': [Q_fordward_reward],
        'greedy_max_reward': [greedy_max_reward],
        'greedy_solution': [greedy_solution],
        'greedy_matchrate': [greedy_matchrate],
        'greedy_tasks_not_serverd': [greedy_tasks_not_serverd],
        'greedy_serverd_rate': [greedy_serverd_rate],
        'greedy_compute_reward': [greedy_compute_reward],
        'greedy_forward_reward': [greedy_forward_reward],
        'random_max_reward': [random_max_reward],
        'random_solution': [random_solution],
        'random_matchrate': [random_matchrate],
        'random_tasks_not_serverd': [random_tasks_not_serverd],
        'random_serverd_rate': [random_serverd_rate],
        'random_compute_reward': [random_compute_reward],
        'random_forward_reward': [random_forward_reward],
        'price': [price],
        'account': [account],
        'computing_resource': [resource],
        'server': [server]
    })

    prefix = 'UAV_dp' if uav_opt else 'UAV_no_dp'
    value = rho if sweep == 'rho' else nu
    filename = f"{prefix}_seed_{seed_number}_{sweep}_{value}.csv"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, filename), index=False, mode='w+', header=True)
    print(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep', choices=['rho', 'nu'], required=True,
                        help='penalty coefficient to sweep: rho (computing, Fig. 3) or nu (communication, Fig. 4)')
    parser.add_argument('--no-uav-opt', dest='uav_opt', action='store_false',
                        help='skip the PSO optimization of the UAV positions')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)),
                        help='random seeds to run (default: 0..9)')
    parser.add_argument('--values', type=float, nargs='+', default=None,
                        help='penalty values to sweep (default: the values used in the paper)')
    parser.add_argument('--out-dir', default=None,
                        help='output directory (default: test_values/<with|no>_UAV_optimize/<sweep>)')
    args = parser.parse_args()

    values = SWEEP_VALUES[args.sweep] if args.values is None else args.values
    out_dir = args.out_dir or os.path.join(ROOT_DIR, 'test_values',
                                           'with_UAV_optimize' if args.uav_opt else 'no_UAV_optimize', args.sweep)

    for seed in args.seeds:
        Initialization(seed, args.sweep, args.uav_opt, values, out_dir)


if __name__ == '__main__':
    main()
