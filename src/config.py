#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Mikhail Kazdagli"
__copyright__ = "Copyright 2021, Symmetry Systems Inc"

import os

# configures the path to IBM CP optimizer
CPO_OPTIMIZER_PATH = "/home/ubuntu/ibm/ILOG/CPLEX_Studio201/cpoptimizer/bin/x86-64_linux/cpoptimizer"

# the public repo contains only synthetic data, thus we run only synthetic experiments
RUN_SYNTHETIC_TESTS = True

# directory containing input data
DATA_DIR = "data"

SYNTHETIC_INPUT_DIR = os.path.join(DATA_DIR, "synthetic_graphs")
REAL_INPUT_DIR = os.path.join(DATA_DIR, "real_graphs")

# output directory containing results of synthetic experiments
SYNTHETIC_RESULTS_DIR = os.path.join(DATA_DIR, "synthetic_results_all_groups")

# output directory containing figures
OUTPUT_FIG_DIR = os.path.join(DATA_DIR, "figs")

# size of user embeddings computed by GNN
EMBEDDING_SIZE = 50

# color constants
COLORS = ["rgb(153, 204, 255)", "rgb(0, 128, 255)", "rgb(0, 76, 153)"]

ORG_ANONYMIZER = {
    # the list of organizations participated in this study is not
    # being publicly shared due to security concerns
}

GLOBAL_PARAMETERS = {
    # limit each CP iteration (Section 3.2) to 15 minutes
    "time_limit_per_run": 15 * 60,
    # limit the total execution time across all iterations (Section 3.2) to 3 hours
    "total_time_limit": 3 * 60 * 60,
    # tolerance of the solution at each iteration (CP's parameter)
    "tolerance": 0.05,
    # tolerance set to control solution improvement with regards to the dark permission reduction
    # the CP solver stops if the current solution is within "tolerance" range of the optimal or
    # if the current solution is within the "dark_permission_tolerance" range of the optimal
    # wrt dark permission component of the objective function
    "dark_permission_tolerance": 0.01,
    # limit the max number of dissimilar users per datastore access group (Section 3.2)
    "max_dissimilar_entity_pairs_per_group": 20,
    # the max number of "separation oracle" iterations (Section 3.2)
    "max_iterations": 100,
    "relative_dark_permission_budget": 0.15,
    # enable user-level homogeneity constraints (Section 3.2)
    "add_user_embedding_constraints": True,
    # enable datastore-level homogeneity constraints; not used in the paper
    "add_data_store_embedding_constraints": False,
    # disable iteratively added user-level homogeneity constraints (Section 3.2)
    # it should be False if we want to iteratively add constraints
    "monolithic_cp_exp": False
}
