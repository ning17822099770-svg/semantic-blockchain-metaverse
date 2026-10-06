# Semantic-Aware Blockchain Metaverse

Python code for the paper **"[Semantic-Aware Blockchain Architecture Design for Edge-enabled Metaverse](https://ieeexplore.ieee.org/abstract/document/10316024)"** by Ning Wang and Beatriz Lorenzo (University of Massachusetts Amherst), *2023 IEEE 14th Annual Ubiquitous Computing, Electronics & Mobile Communication Conference (UEMCON)*, New York, NY, USA, pp. 647–654. DOI: [10.1109/UEMCON59035.2023.10316024](https://doi.org/10.1109/UEMCON59035.2023.10316024)

Metaverse users generate computing tasks with different **semantic types**. Each type has its own price, delay requirement and throughput requirement. A swarm of **UAV edge servers** collects the tasks and runs a **consortium blockchain** that decides which UAV computes each task. The code solves two problems:

| Problem | What is optimized | Method | File |
|---|---|---|---|
| **P1**: UAV deployment | UAV positions that maximize the average A2G + A2A throughput (Eq. 6, 9) | Particle Swarm Optimization (PSO) | `UAV_deployment.py` |
| **P2**: Task allocation | Task-to-server assignment that maximizes blockchain profit under semantic-mismatch penalties (Eq. 7, 8, 10) | Q-learning (ε-greedy), compared with greedy and random selection | `task_allocation.py` |

## Workflow

1. **Task generation and semantic classification.** 60 tasks of K = 4 semantic types are generated, along with 20 UAV servers of the same 4 types.
2. **User–UAV binding.** Each user is bound to its closest UAV, which acts as its trusted server.
3. **UAV deployment (P1).** PSO moves each UAV to maximize the average throughput of its A2G links (to its users) and A2A links (to the other UAVs).
4. **Task allocation (P2).** The blockchain assigns each task to a server for local computing or offloading. Semantic mismatches are penalized by the computing penalty **ρ** and the communication penalty **ν**.
5. **Evaluation.** The total reward, the task–server **matching rate** and the **service rate** (the share of tasks served within their requirements) are recorded for each value of ρ or ν.

## Results

Computing penalty ρ (Fig. 3 in the paper):

| Rewards | Matching rate | Service rate |
|---|---|---|
| ![](figures/rho_1.png) | ![](figures/rho_2.png) | ![](figures/rho_3.png) |

Communication penalty ν (Fig. 4 in the paper):

| Rewards | Matching rate | Service rate |
|---|---|---|
| ![](figures/nu_1.png) | ![](figures/nu_2.png) | ![](figures/nu_3.png) |

Q-learning with PSO-optimized UAV positions gives the highest reward. The reward peaks around ρ = 0.15 and ν = 0.04.

## Repository structure

```
├── run_experiment.py                        # entry point: runs the ρ / ν sweeps and saves CSV results
├── task_generation.py                       # semantic tasks: type, price, data size, delay/throughput requirements, bandwidth
├── server_generation.py                     # UAV servers: type, computing resource, CPU parameters, location, Tx powers
├── binding_and_A2G_communication_model.py   # user–UAV binding and A2G channel rates (Eq. 1–3)
├── A2A_model_get_rate.py                    # A2A channel rates between UAVs (Eq. 4–5)
├── UAV_deployment.py                        # PSO for UAV positioning (P1, Algorithm 1)
├── task_allocation.py                       # rewards, Q-learning, greedy and random allocation (P2, Algorithm 2)
├── figures/                                 # result figures used in the paper
└── test_values/                             # raw simulation results (one CSV per seed and penalty value)
    ├── with_UAV_optimize/{rho,nu}/
    └── no_UAV_optimize/{rho,nu}/
```

## Requirements

- Python 3.9 or 3.10 (the results were produced with Python 3.9)
- `numpy < 2` and `pandas < 2`

```bash
pip install -r requirements.txt
```

`task_allocation.py` updates its tables with chained assignment (`table[col][row] = value`). Under the Copy-on-Write default of pandas 3.x those updates are silently dropped, so use a pandas version below 2.

## Quick start

```bash
# Q-learning with PSO-optimized UAV positions, sweep the computing penalty rho (Fig. 3)
python run_experiment.py --sweep rho

# Same, but sweep the communication penalty nu (Fig. 4)
python run_experiment.py --sweep nu

# Baselines without UAV position optimization
python run_experiment.py --sweep rho --no-uav-opt
python run_experiment.py --sweep nu --no-uav-opt

# A quick run: one seed and two penalty values, written to a custom folder
python run_experiment.py --sweep rho --seeds 0 --values 0 0.15 --out-dir results/
```

The paper averages 100 independent simulations (seeds 0–99, e.g. `--seeds $(seq 0 99)`). By default, seeds 0–9 are run and results are written to `test_values/<with|no>_UAV_optimize/<rho|nu>/` as `UAV_dp_seed_<seed>_<sweep>_<value>.csv` (or `UAV_no_dp_...` without PSO). Each CSV holds one row with the parameters and, for Q-learning, greedy and random selection, the maximum reward, the allocation, the matching rate, the service rate and the unserved tasks.

The PSO step is much slower than the allocation step. Use `--no-uav-opt` for quick checks.

## Key simulation parameters

All values below are the ones set in the code (`run_experiment.py`, `task_generation.py`, `server_generation.py`, the channel models and `UAV_deployment.py`). "type" is the semantic type, 1–4.

**Scenario**

| Parameter | Value in code |
|---|---|
| Map size | 200 m × 200 m |
| Tasks / UAV servers / semantic types | 60 / 20 / 4 (equal number per type) |
| Task data size D_j | uniform integer in [20, 30] Mbit |
| Unit price p_k | 5 · type = 5, 10, 15, 20 |
| Delay requirement per type | 40, 10, 60, 15 |
| Throughput requirement per type | 1, 5, 10, 20 Mbit/s |
| Bandwidth per task type | 5, 25, 50, 100 MHz |
| Server computing resource μ_k | 160 − 20 · type = 140, 120, 100, 80 |
| CPU cycles per unit data θ_k | 2 · type = 2, 4, 6, 8 |
| CPU frequency f_k | 4 · type = 4, 8, 12, 16 |
| CPU architecture parameter α_k | 0.01 · type = 0.01 … 0.04 |
| Transmission / result-return power | uniform in [30, 50] |
| Data-collection power | uniform in [5, 10] |

**Channel**

| Parameter | Value in code |
|---|---|
| Carrier frequency | 2.5 GHz |
| UAV altitude | 50 m |
| Noise power spectral density | 1e-13 |
| A2G channel (urban) a, b, η_LoS, η_NLoS | 9.61, 0.16, 1, 20 |
| A2A channel | free-space path loss with η_LoS |

**PSO (problem P1)**

| Parameter | Value in code |
|---|---|
| Swarm size / iterations | 50 / 100 |
| Inertia weight w, damping w_damp | 1, 0.98 |
| c1, c2 | 1.5, 1.5 |
| Velocity limit | 0.5 × map range |
| Objective weight β between A2A and A2G | 0.5 (fixed as 1/2 and 1/2 in `UAV_deployment.py`) |

The UAVs are optimized one at a time: each PSO run moves one UAV while the others stay where they are.

**Task allocation (problem P2)**

| Parameter | Value in code |
|---|---|
| Learning rate α | 0.9 |
| Discount factor γ | 0.5 |
| ε (probability of the greedy action) | 0.9 |
| Episodes | 500 |
| Intermediary fee ratio λ | 0.1 |
| Mismatched-resource ratio δ | 0.3 |
| Weight of the source server's forwarding reward | 1 |
| ρ sweep / ν sweep | 0–0.95 (step 0.05) / 0–0.09 (step 0.01) |
| Seeds | 0–99 for the paper's results (default 0–9) |

### Differences from Table I of the paper

The results in `test_values/` and the figures were produced by this code. Where the paper's text gives different values, the code values above are the ones that were used:

- **Q-learning α and γ.** The original run scripts passed the hyperparameters to `Task_allocation` in a different order from its signature, so the learning rate was 0.9 and the discount factor 0.5 (Table I lists 0.5 and 0.9). `run_experiment.py` now passes these effective values explicitly. The `ALPHA`/`GAMMA` columns of the CSVs in `test_values/` still show the old labels (0.5/0.9).
- **PSO swarm size.** 50 (Table I lists 100).
- **β.** The PSO objective weights A2A and A2G throughput equally (0.5). `omiga = 0.6` is passed to `Task_allocation` but not used.
- **Computing resources and CPU frequency.** 140/120/100/80 and 4/8/12/16 (the paper lists 120/100/80/60 and 3/6/9/12).
- **Powers.** The transmission and collection powers are drawn from [30, 50] and [5, 10] in the code's units (the paper lists 0.3–0.5 W and 0.05–0.1 W).
- The `price` and `computing_resource` columns of the CSVs are fixed labels, not the values used per task or server.
- NumPy's random generator (used by PSO and the ε-greedy choice) is not seeded, so Q-learning and PSO results vary slightly between runs. Greedy and random selection depend only on the seed and reproduce exactly.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{wang2023semantic,
  author    = {Wang, Ning and Lorenzo, Beatriz},
  title     = {Semantic-Aware Blockchain Architecture Design for Edge-enabled Metaverse},
  booktitle = {2023 IEEE 14th Annual Ubiquitous Computing, Electronics \& Mobile Communication Conference (UEMCON)},
  pages     = {647--654},
  year      = {2023},
  address   = {New York, NY, USA},
  publisher = {IEEE},
  doi       = {10.1109/UEMCON59035.2023.10316024},
  url       = {https://ieeexplore.ieee.org/abstract/document/10316024}
}
```

## Acknowledgements

This work is partially supported by the US National Science Foundation under Grant CNS-2008309.

## License

Released under the MIT License.
