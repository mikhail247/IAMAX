#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Mikhail Kazdagli"
__copyright__ = "Copyright 2021, Symmetry Systems Inc"

import os.path
from glob import glob

import docplex
import joblib
import pandas as pd
from docplex.cp.model import *
from tqdm import tqdm

import config
import src.iam_optimization_cp as iam_optimization_cp
from src.data_loader import Data


def get_solve_status(solution):
    """
    Serves as a generic method to examine CP and MILP solutions.

    Returns "Optimal" if an optimal solution has been found
    """
    if solution is None:
        return "None"

    # if we use MILP solver
    if isinstance(solution, docplex.mp.solution.SolveSolution):
        solve_status = solution.solve_status
        s = solve_status.name
        if s == "OPTIMAL_SOLUTION":
            return "Optimal"
        return "Unknown"

    # if we use CP solver
    return solution.get_solve_status()


def compute_graph_density(data):
    """
    Computes density of a bipartite graph that includes only dynamic accesses to datastores performed by users
    """
    return len(data.dynamic_user_datastore_accesses_set) / (len(data.users) * len(data.data_stores))


def run():
    """
    Runs all experiments and stores results on a disk that can be used later for visualization purposes.
    According to the configuration, it can run experiments with either real or synthetic data.
    """

    # directories where to load datasets from
    org_dirs = list()
    # specify the number of data access groups to use (Section 3)
    group_count_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # store graph densities
    densities = list()

    if config.RUN_SYNTHETIC_TESTS:
        print("Processing synthetic graphs")

        for folder in glob(os.path.join(config.SYNTHETIC_INPUT_DIR, "*")):
            if os.path.isdir(folder):
                org_dirs.append(folder)
    else:
        print("Processing real graphs")

        for org in config.ORG_ANONYMIZER.keys():
            org_dir = os.path.join(config.DATA_DIR, org)
            org_dirs.append(org_dir)

    # determine the output directory
    if len(group_count_values) > 1:
        output_dir = "results_all_groups"
    else:
        output_dir = "results_group_{}".format(group_count_values[0])

    # add prefix to the output directory when experimenting with synthetic data
    if config.RUN_SYNTHETIC_TESTS:
        output_dir = "synthetic_{}".format(output_dir)

    output_dir = os.path.join(config.DATA_DIR, output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # iterate over all data sets
    for i, org_dir in tqdm(enumerate(org_dirs)):
        org = os.path.basename(org_dir)
        print("{}. Organization: {} {}".format(i, org, config.ORG_ANONYMIZER.get(org, "unknown")))
        # keep intermediate results in the "results" dictionary
        results = dict()
        output_file_name = os.path.join(output_dir, "results_{}.joblib".format(org))

        # load partially computed results
        if os.path.exists(output_file_name):
            results = joblib.load(output_file_name)

        if not os.path.exists(org_dir):
            continue

        # skip solving an already solved problem
        if len(group_count_values) == 1 and os.path.exists(output_file_name):
            group_count = group_count_values[0]
            baseline_solution = results[str(group_count)]["solution_stats"][0]
            total_time = baseline_solution["process_infos"]["TotalTime"]
            print("Organization: {} TotalTime: {}".format(org, total_time))
            continue

        # load data
        data = Data(org_dir, config.GLOBAL_PARAMETERS)
        density = compute_graph_density(data)
        densities.append(density)
        print("{} Organization: {} Graph density: {}".format(i, org, density))

        # iterate over all group counts
        for group_count in group_count_values:
            data.set_group_count(group_count)

            if str(group_count) in results:
                continue

            results.setdefault(str(group_count), dict())
            print("Processing organization {} with group_count = {}".format(org, group_count))
            # make a local copy of parameters
            parameters = copy.deepcopy(config.GLOBAL_PARAMETERS)
            parameters.update({
                "org": org,
                "org_dir": org_dir,
                "group_count": group_count,
                "dissimilarity_threshold": data.max_user_cluster_diameter,
                "output_file_name": output_file_name
            })

            start_time = time.time()
            solution, stats = iam_optimization_cp.run_experiments(data, parameters, results)
            print("Execution time: CP solver: {} sec".format(time.time() - start_time))
            print("Objective value", solution.get_objective_value())
            stats.pop("last_feasible_solution", None)
            objective_value = None

            if get_solve_status(solution) in {"Optimal", "Feasible"}:
                solution.model.data = None
                objective_value = solution.get_objective_value()

            relative_dark_permissions_fraction = stats["last_solution_stats"]["relative_dark_permissions_fraction"]
            print("[{}] Objective value: {} group_count: {}, relative_dark_permissions_fraction: {}".format(org,
                                                                                                            objective_value,
                                                                                                            group_count,
                                                                                                            relative_dark_permissions_fraction))

            stats["org"] = org
            results[str(group_count)] = stats
            # store results on a disk
            joblib.dump(results, output_file_name)


def dump_solution_stats(synthetic_graphs, group="20"):
    """
    Report basic parameters of the CP program such the number of variables, constraints, max memory allocated, etc
    """
    stats = list()
    selected_orgs = config.ORG_ANONYMIZER.keys()
    dir_name = os.path.join(config.DATA_DIR, "results_all_groups")
    output_dir_name = os.path.join(config.DATA_DIR, "solution_stats")

    if not os.path.exists(output_dir_name):
        os.mkdir(output_dir_name)

    file_name_results_list = list()
    synth_dir = os.path.join(config.DATA_DIR, "synthetic_results_group_20")

    if synthetic_graphs:
        output_file_name = "synth_graph_stats.csv"

        for file in glob(os.path.join(synth_dir, "*")):
            if not os.path.isdir(file):
                org = "synth"
                file_name_results_list.append((org, file))
    else:
        output_file_name = "real_graph_stats.csv"

        for org in selected_orgs:
            file_name_results = os.path.join(dir_name, "results_{}.joblib".format(org))
            file_name_results_list.append((org, file_name_results))

    for org, file_name_results in tqdm(file_name_results_list):
        print(file_name_results)

        if not os.path.exists(file_name_results):
            continue

        cp_solution = joblib.load(file_name_results)
        solution = cp_solution[group]["last_solution_stats"]
        peak_memory_usage_gd = solution["process_infos"]["PeakMemoryUsage"] / (1024 * 1024 * 1024)
        memory_usage_gd = solution["process_infos"]["MemoryUsage"] / (1024 * 1024 * 1024)

        stats.append({
            "organization": org,
            "anonymized_organization": config.ORG_ANONYMIZER.get(org, ""),
            "# of variables": solution["process_infos"]["NumberOfVariables"],
            "# of constraints": solution["process_infos"]["NumberOfConstraints"],
            "peak memory usage, GB": peak_memory_usage_gd,
            "memory usage, GB": memory_usage_gd
        })

        df = pd.DataFrame(stats)
        file_name = os.path.join(output_dir_name, output_file_name)
        df.to_csv(file_name)
        file_name = os.path.join(output_dir_name, "describe_" + output_file_name)
        df.describe().to_csv(file_name)


if __name__ == "__main__":
    """
    Run all experiments
    """
    run()
