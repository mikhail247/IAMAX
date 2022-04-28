#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Mikhail Kazdagli"
__copyright__ = "Copyright 2021, Symmetry Systems Inc"

"""
A module to visualize results using Plotly library.
"""

from collections import Counter
from glob import glob

import joblib
import builtins
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats
from docplex.cp.model import *

import config
import src.iam_optimization_cp as iam_optimization_cp
from src.data_loader import Data


def parse_variables(all_var_solutions):
    """
    Parse the CP solution
    """
    user_groups = dict()
    data_store_groups = dict()

    for variable in all_var_solutions:
        var_name = variable.get_name()
        var_value = variable.get_value()

        if "user" in var_name and "group" in var_name and var_value != 0:
            user, group = var_name.split("--")
            user_groups.setdefault(group, set())
            user_groups[group].add(user)

        if "ds" in var_name and "group" in var_name and var_value != 0:
            group, ds = var_name.split("--")
            data_store_groups.setdefault(group, set())
            data_store_groups[group].add(ds)

    return user_groups, data_store_groups


def get_satisfied_constraints(solution_stats, violated_user_objects, violated_data_store_objects):
    """
    Compute the number of (un)satisfied constraints at every iteration.
    """
    results = list()

    for iter in range(len(solution_stats)):
        stats = solution_stats[iter]

        if stats["solve_status"] not in {"Optimal", "Feasible"}:
            if iter == 0:
                results.append((len(violated_user_objects), len(violated_data_store_objects)))
            break

        if "vars" not in stats:
            break

        all_var_solutions = stats["vars"]
        user_groups, data_store_groups = parse_variables(all_var_solutions)
        violated_user_constraints = 0
        violated_data_store_constraints = 0

        for user_a, user_b in violated_user_objects:
            for group, group_entities in user_groups.items():
                if user_a in group_entities and user_b in group_entities:
                    violated_user_constraints += 1
                    break

        for ds_a, ds_b in violated_data_store_objects:
            for group, group_entities in data_store_groups.items():
                if ds_a in group_entities and ds_b in group_entities:
                    violated_data_store_constraints += 1
                    break

        results.append((violated_user_constraints, violated_data_store_constraints))
    return results


def get_filtered_vars(all_var_solutions, below_cutoff_users):
    """
    Filter variables
    """
    filtered_all_var_solutions = list()

    if all_var_solutions is None:
        return None

    for var in all_var_solutions:
        var_name = var.get_name()

        if "user" in var_name and "ds" in var_name:
            user = var_name.split("--")[0]

            if user in below_cutoff_users:
                filtered_all_var_solutions.append(var)

    return filtered_all_var_solutions


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def compute_entropy(static_data, all_var_solutions):
    """
    Compute entropy
    """
    if all_var_solutions is None:
        return None

    solution_groups = dict()
    group_entropy = dict()

    for var in all_var_solutions:
        var_name = var.get_name()
        var_value = var.get_value()

        if "user" in var_name and "group" in var_name and var_value != 0:
            user, group = var_name.split("--")
            cluster_id = static_data.get_cluster_id(user)
            solution_groups.setdefault(group, Counter())
            solution_groups[group].update([cluster_id])

    for group, counter in solution_groups.items():
        freq = list(dict(counter).values())
        group_entropy[group] = (scipy.stats.entropy(freq), len(counter))

    total_sample_count = sum([x[1] for x in group_entropy.values()])
    entropy = sum([entropy * sample_count / total_sample_count for entropy, sample_count in group_entropy.values()])
    return entropy


def get_last_solution(solution, rel_dark_perm_upper_bound):
    """
    Extract the last feasible solution while using the constraints generation approach
    """
    solution_stats = solution["solution_stats"]
    solution_idx = 0
    last_solution = solution_stats[0]

    for iter in range(len(solution_stats)):
        iter_data = solution_stats[iter]
        relative_dark_permissions_fraction = iter_data["relative_dark_permissions_fraction"]

        if iter_data["solve_status"] not in {"Optimal", "Feasible"}:
            break

        if iter > 0 and (
                relative_dark_permissions_fraction is None or relative_dark_permissions_fraction > rel_dark_perm_upper_bound):
            pass
        else:
            solution_idx = iter
            last_solution = solution_stats[solution_idx]

    return last_solution, solution_idx


def visualize_gnn_embeddings(org_dir):
    """
    Visualize GNN embeddings (Fig. 3a)
    """
    input_dir = os.path.join(config.DATA_DIR, org_dir)
    file_name = os.path.join(input_dir, "source_identity_clusters.parquet")

    if not os.path.exists(file_name):
        if config.RUN_SYNTHETIC_TESTS:
            print("embeddings visualization is not available for synthetic graphs")

        print("{} is not found".format(file_name))
        return

    df = pd.read_parquet(file_name)
    fig = go.Figure()

    for cluster_id in df["cluster_id"].unique():
        df_cluster = df[df["cluster_id"] == cluster_id]
        dot_coordinates = np.vstack(df_cluster["2d"].tolist())

        fig.add_trace(go.Scatter(
            x=dot_coordinates[:, 0],
            y=dot_coordinates[:, 1],
            mode="markers",
            text=df_cluster["node_label"]))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,

        xaxis=dict(
            title_text="Component 1",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along X axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            zeroline=False,
            zerolinecolor="grey",
            zerolinewidth=0.5
        ),
        yaxis=dict(
            title_text="Component 2",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along Y axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            zeroline=False,
            zerolinecolor="grey",
            zerolinewidth=0.5
        ),
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1.0,
            bgcolor="#FFFFFF",
            font=dict(size=15)
        )
    )

    output_dir_name = os.path.join(config.DATA_DIR, "figs")
    html_file_path = os.path.join(output_dir_name, "gnn_embeddings_visualized")
    fig.write_html(html_file_path + ".html")
    fig.write_image(html_file_path + ".eps")
    fig.show()
    return fig


def plot_relative_dark_permissions_fraction_vs_group_count():
    """
    Relative percentage of remaining dark permissions after IAM policy optimization as a function of the number of
    generated data access groups (Fig. 3b)
    """

    if config.RUN_SYNTHETIC_TESTS:
        file_names = list()

        for file_name in glob(os.path.join(config.SYNTHETIC_RESULTS_DIR, "*")):
            temp = os.path.basename(file_name).split(".")[0]
            org = temp[len("results_"):]
            file_names.append((org, file_name))
    else:
        dir_name = os.path.join(config.DATA_DIR, "results_all_groups")
        file_names = [(org, os.path.join(dir_name, "results_{}.joblib".format(org))) for org in config.ORG_ANONYMIZER.keys()]

    fig = go.Figure()

    for org, file_name in file_names:
        print(file_name)

        if not os.path.exists(file_name):
            continue

        data = joblib.load(file_name)
        group_counts = sorted([int(x) for x in data.keys()])
        y_values = list()

        for group_count in group_counts:
            relative_dark_permissions_fraction = data[str(group_count)]["solution_stats"][0][
                "relative_dark_permissions_fraction"]
            if relative_dark_permissions_fraction is not None:
                relative_dark_permissions_fraction *= 100
            y_values.append(relative_dark_permissions_fraction)

        anonymized_org_name = config.ORG_ANONYMIZER.get(org, org)
        label = "Data set {}".format(anonymized_org_name.split("_")[1])
        fig.add_trace(go.Scatter(x=group_counts, y=y_values, mode="lines", name=label))

    fig.update_layout(
        xaxis_range=[0, 30],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        xaxis=dict(
            title_text="Group count",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along X axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="grey",
            zerolinewidth=0.5
        ),
        yaxis=dict(
            title_text="Dark permissions, %",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along Y axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="grey",
            zerolinewidth=0.5
        ),
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1.0,
            bgcolor="#FFFFFF",
            font=dict(size=15)
        )
    )

    html_file_path = os.path.join(config.OUTPUT_FIG_DIR, "rel_dark_perm_vs_group_count_no_embeddings")
    fig.write_html(html_file_path + ".html")
    fig.write_image(html_file_path + ".eps")
    fig.show()
    return fig


def plot_unsatisfied_embedding_constraints_vs_iteration(group_count, rel_dark_perm_upper_bound=0.05):
    """
    Visualize cumulative fraction of satisfied user embedding constraints at each iteration (Fig. 4a)
    """
    group_count = str(group_count)
    fig = go.Figure()

    if config.RUN_SYNTHETIC_TESTS:
        file_names = list()

        for file_name in glob(os.path.join(config.SYNTHETIC_RESULTS_DIR, "*")):
            temp = os.path.basename(file_name).split(".")[0]
            org = temp[len("results_"):]
            file_names.append((org, file_name))
    else:
        dir_name = os.path.join(config.DATA_DIR, "results_all_groups")
        file_names = [(org, os.path.join(dir_name, "results_{}.joblib".format(org))) for org in config.ORG_ANONYMIZER.keys()]

    for org, file_name in file_names:
        print(org)

        if config.RUN_SYNTHETIC_TESTS:
            org_dir = os.path.join(config.SYNTHETIC_INPUT_DIR, org)
        else:
            org_dir = os.path.join(config.REAL_INPUT_DIR, org)

        static_data = Data(org_dir, config.GLOBAL_PARAMETERS, log_level=0)
        # get the total number of violated constraints according to the clustering results
        violated_user_objects, violated_data_store_objects = static_data.get_violated_constraints(
            config.GLOBAL_PARAMETERS,
            static_data.max_user_cluster_diameter)
        total_violated_constraints = len(violated_user_objects) + len(violated_data_store_objects)
        solution_data = joblib.load(file_name)

        # select solution stats for a given group_count
        solution_stats = solution_data[group_count]["solution_stats"]
        violated_constraints_count_per_iteration = get_satisfied_constraints(solution_stats, violated_user_objects,
                                                                             violated_data_store_objects)

        # add '1.0" point at the beginning
        y_values = [1.0]
        iterations = np.arange(start=0, stop=len(violated_constraints_count_per_iteration))

        for iter in iterations:
            violated_constraints = sum(violated_constraints_count_per_iteration[iter])
            y_values.append(violated_constraints / total_violated_constraints)

        anonymized_org_name = config.ORG_ANONYMIZER.get(org, org)
        label = "Data set {}".format(anonymized_org_name.split("_")[1])

        iterations = np.arange(start=0, stop=len(violated_constraints_count_per_iteration) + 1)
        fig.add_trace(go.Scatter(
            x=iterations, y=100 * np.array(y_values), mode="lines", name=label,
            line=dict(width=2)
        ))

    fig.update_layout(
        yaxis_range=[0, 105],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        xaxis=dict(
            title_text="Iteration",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along X axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="grey",
            zerolinewidth=0.5
        ),
        yaxis=dict(
            title_text="Unsatisfied constraints, %",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along Y axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="grey",
            zerolinewidth=0.5
        ),
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.85,
            bgcolor="#FFFFFF",
            font=dict(size=15)
        )
    )

    html_file_path = os.path.join(config.OUTPUT_FIG_DIR, "unsatisfied_embedding_constraints_per_iteration")
    fig.write_html(html_file_path + ".html")
    fig.write_image(html_file_path + ".eps")
    fig.show()
    return fig


def plot_unsatisfied_embedding_constraints_vs_group_count(rel_dark_perm_upper_bound=0.05):
    """
    Visualize the number of unsatisfied embedding constraints vs the number of groups.
    """
    if config.RUN_SYNTHETIC_TESTS:
        file_names = list()

        for file_name in glob(os.path.join(config.SYNTHETIC_RESULTS_DIR, "*")):
            temp = os.path.basename(file_name).split(".")[0]
            org = temp[len("results_"):]
            file_names.append((org, file_name))
    else:
        dir_name = os.path.join(config.DATA_DIR, "results_all_groups")
        file_names = [(org, os.path.join(dir_name, "results_{}.joblib".format(org))) for org in
                      config.ORG_ANONYMIZER.keys()]

    fig = go.Figure()

    for org, file_name in file_names:
        print(org)

        if config.RUN_SYNTHETIC_TESTS:
            org_dir = os.path.join(config.SYNTHETIC_INPUT_DIR, org)
        else:
            org_dir = os.path.join(config.REAL_INPUT_DIR, org)

        static_data = Data(org_dir, config.GLOBAL_PARAMETERS, log_level=0)
        violated_user_objects, violated_data_store_objects = static_data.get_violated_constraints(
            config.GLOBAL_PARAMETERS,
            static_data.max_user_cluster_diameter)
        total_violated_constraints = len(violated_user_objects) + len(violated_data_store_objects)

        solution_data = joblib.load(file_name)
        violated_constraints_count = dict()

        for group_count, obj in solution_data.items():
            solution_stats = solution_data[group_count]["solution_stats"]
            violated_constraints_count[group_count] = get_satisfied_constraints(solution_stats, violated_user_objects,
                                                                                violated_data_store_objects)

        group_counts = sorted([int(x) for x in solution_data.keys()])
        y_values = list()

        for group_count in group_counts:
            _, solution_idx = get_last_solution(solution_data[str(group_count)], rel_dark_perm_upper_bound)
            violated_constraints = sum(violated_constraints_count[str(group_count)][solution_idx])
            y_values.append(violated_constraints / total_violated_constraints)

        anonymized_org_name = config.ORG_ANONYMIZER.get(org, org)
        fig.add_trace(go.Scatter(x=group_counts, y=y_values, mode="lines", name=anonymized_org_name))

    fig.update_layout(
        title="Relative fraction of unsatisfied embedding constraints vs group count",
        xaxis_title="# of groups",
        yaxis_title="fraction of unsatisfied embedding constraints",
    )
    html_file_path = os.path.join(config.OUTPUT_FIG_DIR, "unsatisfied_embedding_constraints.html")
    fig.write_html(html_file_path)
    fig.show()


def plot_max_attack_impact(max_compromised_users, group_count, percentile_cut_off=80,
                           rel_dark_perm_upper_bound=0.05):
    """
    Visualize the worst-case attack impact on an organization's infrastructure (Fig. 4d)
    """
    group_count = str(group_count)

    if config.RUN_SYNTHETIC_TESTS:
        file_names = list()

        for file_name in glob(os.path.join(config.SYNTHETIC_RESULTS_DIR, "*")):
            temp = os.path.basename(file_name).split(".")[0]
            org = temp[len("results_"):]
            file_names.append((org, file_name))
    else:
        dir_name = os.path.join(config.DATA_DIR, "results_all_groups")
        file_names = [(org, os.path.join(dir_name, "results_{}.joblib".format(org))) for org in
                      config.ORG_ANONYMIZER.keys()]

    y_values = dict()
    fig = go.Figure()

    for org, file_name in file_names:
        print(org)

        if config.RUN_SYNTHETIC_TESTS:
            org_dir = os.path.join(config.SYNTHETIC_INPUT_DIR, org)
        else:
            org_dir = os.path.join(config.REAL_INPUT_DIR, org)

        if not os.path.exists(file_name) or not os.path.exists(org_dir):
            continue

        data = Data(org_dir, config.GLOBAL_PARAMETERS, log_level=0)
        dynamic_data = joblib.load(file_name)

        # remove users with large degree
        user_degree = dict()
        for user, ds in data.dynamic_user_datastore_accesses_set:
            user_degree.setdefault(user, set())
            user_degree[user].add(ds)

        user_degree = {k: len(v) for k, v in user_degree.items()}
        cutoff_value = np.percentile(list(user_degree.values()), percentile_cut_off)
        # final set of users
        below_cutoff_users = set([user for user, val in user_degree.items() if val < cutoff_value])

        # compute max impact before optimization
        user_datastore_permissions_set = {(user, ds) for user, ds in data.user_datastore_permissions_set if
                                          user in below_cutoff_users}

        for k_users in range(1, max_compromised_users + 1):
            original_selected_users, original_compromised_data_stores = iam_optimization_cp.get_max_k_impact(
                k=k_users,
                sparse_adj_matrix=user_datastore_permissions_set)
            max_impact_before_optimization = (len(original_compromised_data_stores))

            all_var_solutions = dynamic_data[str(group_count)]["solution_stats"][0].get("vars", None)
            # evaluate impact after optimization without embedding constraints
            filtered_all_var_solutions = get_filtered_vars(all_var_solutions, below_cutoff_users)
            selected_users_opt_no_embeddings, compromised_data_stores_opt_no_embeddings, cluster = iam_optimization_cp.get_max_k_impact_wrt_clsuters(
                k=k_users, user_clusters=data.get_user_clusters(),
                sparse_adj_matrix=iam_optimization_cp.get_solution_user_data_store_edges(
                    solution=None,
                    all_var_solutions=filtered_all_var_solutions))
            max_impact_opt_no_embeddings = len(compromised_data_stores_opt_no_embeddings)

            # evaluate impact after an optimization with embedding constraints
            last_solution_stats, _ = get_last_solution(dynamic_data[str(group_count)], rel_dark_perm_upper_bound)
            all_var_solutions = last_solution_stats.get("vars", None)
            filtered_all_var_solutions = get_filtered_vars(all_var_solutions, below_cutoff_users)
            selected_users_opt_embeddings, compromised_data_stores_opt_embeddings, cluster = iam_optimization_cp.get_max_k_impact_wrt_clsuters(
                k=k_users, user_clusters=data.get_user_clusters(),
                sparse_adj_matrix=iam_optimization_cp.get_solution_user_data_store_edges(
                    solution=None,
                    all_var_solutions=filtered_all_var_solutions))
            max_impact_opt_with_embeddings = len(compromised_data_stores_opt_embeddings)

            anonymized_org_name = config.ORG_ANONYMIZER.get(org, org)
            y_values.setdefault(anonymized_org_name, list())
            y_values[anonymized_org_name].append(
                100 * max_impact_opt_with_embeddings / (max_impact_before_optimization + 1E-6))

    legend = ["1 compromised user", "2 compromised users", "3 compromised users"]
    x_labels = [x.split("_")[1] for x in y_values.keys()]
    for k_users in range(1, max_compromised_users + 1):
        fig.add_trace(go.Bar(x=x_labels,
                             y=[x[k_users - 1] for x in y_values.values()],
                             name=legend[k_users - 1],
                             marker_color=config.COLORS[k_users - 1]
                             ))

    fig.update_layout(
        yaxis_range=[0, 105],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        xaxis=dict(
            title_text="Data sets",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along X axes
        ),
        yaxis=dict(
            title_text="Relative attack impact, %",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along Y axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1
        ),
        legend=dict(
            yanchor="top",
            y=1.04,
            xanchor="left",
            x=0.78,
            bgcolor="#FFFFFF",
            font=dict(size=15)
        )
    )

    html_file_path = os.path.join(config.OUTPUT_FIG_DIR,
                                  "max_relative_impact_of_compromising_{}_users".format(max_compromised_users))
    fig.write_html(html_file_path + ".html")
    fig.write_image(html_file_path + ".eps")
    fig.show()
    return fig


def plot_avg_attack_impact(max_compromised_users, group_count, rel_dark_perm_upper_bound=0.05):
    """
    Visualize the average-case attack impact on an organization's infrastructure (Fig. 4c)
    """
    group_count = str(group_count)

    if config.RUN_SYNTHETIC_TESTS:
        file_names = list()

        for file_name in glob(os.path.join(config.SYNTHETIC_RESULTS_DIR, "*")):
            temp = os.path.basename(file_name).split(".")[0]
            org = temp[len("results_"):]
            file_names.append((org, file_name))
    else:
        dir_name = os.path.join(config.DATA_DIR, "results_all_groups")
        file_names = [(org, os.path.join(dir_name, "results_{}.joblib".format(org))) for org in
                      config.ORG_ANONYMIZER.keys()]

    legend = ["1 compromised user", "2 compromised users", "3 compromised users"]
    y_values = dict()
    fig = go.Figure()

    for org, file_name in file_names:
        print(org, group_count)

        if config.RUN_SYNTHETIC_TESTS:
            org_dir = os.path.join(config.SYNTHETIC_INPUT_DIR, org)
        else:
            org_dir = os.path.join(config.REAL_INPUT_DIR, org)

        if not os.path.exists(file_name) or not os.path.exists(org_dir):
            continue

        dynamic_data = joblib.load(file_name)
        last_solution_stats, _ = get_last_solution(dynamic_data[str(group_count)], rel_dark_perm_upper_bound)
        solution_status = last_solution_stats["solve_status"]
        all_var_solutions = last_solution_stats.get("vars", None)
        all_var_solutions_dict = {x.get_name(): x for x in all_var_solutions}

        data = Data(org_dir, config.GLOBAL_PARAMETERS, all_var_solutions=all_var_solutions, log_level=0)

        for k_users in range(1, max_compromised_users + 1):
            attack_stats_before_opt = iam_optimization_cp.estimate_malicious_impact_before_optimization(data, k_users)

            # filtered_all_var_solutions = get_filtered_vars(all_var_solutions, below_cutoff_users)
            attack_stats_after_opt = iam_optimization_cp.estimate_malicious_impact_after_optimization(solution_status,
                                                                                                      all_var_solutions_dict,
                                                                                                      data, k_users)
            anonymized_org_name = config.ORG_ANONYMIZER.get(org, org)
            y_values.setdefault(anonymized_org_name, list())
            y_values[anonymized_org_name].append(
                100 * attack_stats_after_opt["mean"] / attack_stats_before_opt["mean"])
            print(attack_stats_before_opt["mean"], attack_stats_after_opt["mean"],
                  attack_stats_after_opt["mean"] / attack_stats_before_opt["mean"])

    x_labels = [x.split("_")[1] for x in y_values.keys()]
    for k_users in range(1, max_compromised_users + 1):
        fig.add_trace(go.Bar(x=x_labels,
                             y=[x[k_users - 1] for x in y_values.values()],
                             name=legend[k_users - 1],
                             marker_color=config.COLORS[k_users - 1]))

    fig.update_layout(
        yaxis_range=[0, 105],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        xaxis=dict(
            title_text="Data sets",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along X axes
        ),
        yaxis=dict(
            title_text="Relative attack impact, %",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along Y axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1
        ),
        legend=dict(
            yanchor="top",
            y=1.04,
            xanchor="left",
            x=0.78,
            bgcolor="#FFFFFF",
            font=dict(size=15)
        )
    )

    html_file_path = os.path.join(config.OUTPUT_FIG_DIR,
                                  "avg_relative_impact_of_compromising_{}_users".format(max_compromised_users))
    fig.write_html(html_file_path + ".html")
    fig.write_image(html_file_path + ".eps")
    fig.show()
    return fig


def plot_entropy_reduction_with_embedding_constraints(group_count=20, rel_dark_perm_upper_bound=0.05):
    """
    Visualize reduction of entropy across data access groups after adding embedding constraints (Fig. 4b)
    """
    group_count = str(group_count)

    if config.RUN_SYNTHETIC_TESTS:
        file_names = list()

        for file_name in glob(os.path.join(config.SYNTHETIC_RESULTS_DIR, "*")):
            temp = os.path.basename(file_name).split(".")[0]
            org = temp[len("results_"):]
            file_names.append((org, file_name))
    else:
        dir_name = os.path.join(config.DATA_DIR, "results_all_groups")
        file_names = [(org, os.path.join(dir_name, "results_{}.joblib".format(org))) for org in config.ORG_ANONYMIZER.keys()]

    fig = go.Figure()
    entropy_values = dict()

    for org, file_name in file_names:
        print(org)

        if config.RUN_SYNTHETIC_TESTS:
            org_dir = os.path.join(config.SYNTHETIC_INPUT_DIR, org)
        else:
            org_dir = os.path.join(config.REAL_INPUT_DIR, org)

        if not os.path.exists(file_name) or not os.path.exists(org_dir):
            continue

        data = Data(org_dir, config.GLOBAL_PARAMETERS, log_level=0)
        dynamic_data = joblib.load(file_name)

        # get solution without embedding cosntraints
        all_var_solutions = dynamic_data[str(group_count)]["solution_stats"][0].get("vars", None)
        entropy_no_embedding_constraints = compute_entropy(data, all_var_solutions)

        # get solution after adding embedding constraints
        last_solution_stats, _ = get_last_solution(dynamic_data[str(group_count)], rel_dark_perm_upper_bound)
        all_var_solutions = last_solution_stats.get("vars", None)
        entropy_with_embedding_constraints = compute_entropy(data, all_var_solutions)

        # save to a dictionary
        anonymized_org_name = config.ORG_ANONYMIZER.get(org, org)
        entropy_values[anonymized_org_name] = (
            entropy_no_embedding_constraints, entropy_with_embedding_constraints)
        print(entropy_no_embedding_constraints, entropy_with_embedding_constraints)

    legend = ["Without embedding constraints", "With embedding constraint"]
    x_labels = [x.split("_")[1] for x in entropy_values.keys()]
    bar_values = [
        [builtins.round(x[k], 2) for x in entropy_values.values()]
        for k in range(2)]

    for k in range(2):
        fig.add_trace(go.Bar(x=x_labels,
                             y=bar_values[k],
                             text=bar_values[k],
                             textfont={"size": 15},
                             textposition="outside",
                             name=legend[k],
                             marker_color=config.COLORS[k]
                             ))

    y_max = max(max(bar_values[0]), max(bar_values[1]))

    fig.update_layout(
        yaxis_range=[0, 1.1 * y_max],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        xaxis=dict(
            title_text="Data sets",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along X axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="grey",
            zerolinewidth=0.5
        ),
        yaxis=dict(
            title_text="Entropy",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along Y axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="grey",
            zerolinewidth=0.5
        ),
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.01,
            bgcolor="#FFFFFF",
            font=dict(size=15)
        )
    )

    html_file_path = os.path.join(config.OUTPUT_FIG_DIR, "entropy_group_count_{}".format(group_count))
    fig.write_html(html_file_path + ".html")
    fig.write_image(html_file_path + ".eps")
    fig.show()
    return fig


def plot_synthetic_results(group_count=20):
    """
    Visualize distribution of the fraction of remaining dark permissions (Fig. 3c)
    """
    resuts_file = list()
    group_count = str(group_count)

    for file in glob(os.path.join(config.SYNTHETIC_RESULTS_DIR, "*")):
        if os.path.isfile(file):
            resuts_file.append(file)

    y_values = list()

    for i, file in enumerate(resuts_file):
        data = joblib.load(file)
        relative_dark_permissions_fraction = data[str(group_count)]["solution_stats"][0][
            "relative_dark_permissions_fraction"]

        if relative_dark_permissions_fraction is not None:
            relative_dark_permissions_fraction *= 100

        y_values.append(relative_dark_permissions_fraction)

    edges = OrderedDict()
    for k in range(0, 100, 5):
        edges[k] = 0

    for val in y_values:
        for k in edges.keys():
            if val <= k:
                edges[k] += 1
                break

    x_range = max([k for k, v in edges.items() if v > 0])
    bar_labels = [x if x else None for x in edges.values()]
    x = [x - 2.5 for x in edges.keys()]
    width = [5] * len(edges)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x,
                         y=list(edges.values()),
                         text=bar_labels,
                         textfont={"size": 15},
                         textposition="outside",
                         marker_color=config.COLORS[1],
                         width=width
                         ))

    y_max = max(edges.values())

    fig.update_layout(
        xaxis_range=[-4.8, x_range + 3],
        yaxis_range=[0, 1.1 * y_max],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        xaxis=dict(
            title_text="Data sets",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along X axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="grey",
            zerolinewidth=0.5
        ),
        yaxis=dict(
            title_text="Entropy",
            title_font={"size": 20},
            title_standoff=25,
            tickfont=dict(size=20),  # font size along Y axes
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="grey",
            zerolinewidth=0.5
        ),
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.01,
            bgcolor="#FFFFFF",
            font=dict(size=15)
        )
    )

    html_file_path = os.path.join(config.OUTPUT_FIG_DIR,"synthetic_graph_experiment_no_embeddings_constraints_group_count_{}".format(group_count))
    fig.write_html(html_file_path + ".html")
    fig.write_image(html_file_path + ".eps")
    fig.show()
    return fig


if __name__ == "__main__":
    """
    Generate plots used in the paper.
    """
    # must be replaced with the name of the actual dataset, e.g. "graph_15"
    org_name = "<organization>"
    make_dir(config.OUTPUT_FIG_DIR)
    plot_synthetic_results()
    visualize_gnn_embeddings(org_name)
    plot_relative_dark_permissions_fraction_vs_group_count()
    plot_entropy_reduction_with_embedding_constraints(group_count=20, rel_dark_perm_upper_bound=0.05)
    plot_unsatisfied_embedding_constraints_vs_iteration(group_count=20, rel_dark_perm_upper_bound=0.05)
    plot_max_attack_impact(max_compromised_users=3, group_count=20, percentile_cut_off=80,
                           rel_dark_perm_upper_bound=0.05)
    plot_avg_attack_impact(max_compromised_users=3, group_count=20, rel_dark_perm_upper_bound=0.05)
    plot_unsatisfied_embedding_constraints_vs_group_count()
