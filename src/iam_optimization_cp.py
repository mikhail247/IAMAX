#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Mikhail Kazdagli"
__copyright__ = "Copyright 2021, Symmetry Systems Inc"

import datetime
import random
import shutil
import statistics

import docplex
import joblib
import numpy as np
from docplex.cp.model import *
from tqdm import tqdm

import config
from src.callbacks import MonitorRelativeDarkPermissionReduction


def add_variables(model, *args, **kwargs):
    """
    Add model variables
    """

    # replace some variables with constants if we don't need to optimize them
    constants = kwargs.get("constant_vars", {})

    # add variables UG (Section 3, "Users and datastore access groups")
    model.user_data_access_group = dict()
    for user in model.data.get_users():
        for group in model.data.get_groups():
            key = (user, group)
            if key in constants:
                model.user_data_access_group[key] = constants[key]
            else:
                model.user_data_access_group[key] = model.binary_var(name="{}--{}".format(user, group))

    # add variables GD (Section 3, "Datastore access groups and datastores")
    model.data_access_group_datastore = dict()
    for group in model.data.get_groups():
        for data_store in model.data.get_data_stores():
            key = (group, data_store)
            if key in constants:
                model.data_access_group_datastore[key] = constants[key]
            else:
                model.data_access_group_datastore[key] = model.binary_var(
                    name="{}--{}".format(group, data_store))

    # add variables $\widetilde{\ud}(u, d)$} (Section 3, "Frequently accessed datastores must still be accessible")
    model.user_datastore = dict()
    for user in model.data.get_users():
        for data_store in model.data.get_data_stores():
            key = (user, data_store)
            if key in constants:
                model.user_datastore[key] = constants[key]
            else:
                model.user_datastore[key] = model.binary_var(name="{}--{}".format(user, data_store))


def add_constraints(model, *args, **kwargs):
    """
    Add model constraints
    """

    # usually, it's set to False
    monolithic_cp_exp = kwargs["monolithic_cp_exp"]

    def allow_dynamic_access(user, data_store):
        return model.user_datastore[user, data_store] >= model.data.get_dynamic_user_datastore_accesses(user,
                                                                                                        data_store)

    def user_group_data_store_path(user, group, data_store):
        a = model.user_data_access_group[user, group]
        b = model.data_access_group_datastore[group, data_store]
        return model.sum([a, b]) == 2

    def user_datastore_path_exists(user, data_store):
        paths = [user_group_data_store_path(user, group, data_store) for group in model.data.get_groups()]
        return model.sum(paths) >= 1

    def compute_datastore_access_groups(user, data_store):
        path_component = model.sum([user_datastore_path_exists(user, data_store),
                                    model.data.get_user_datastore_permissions(user, data_store)]) == 2
        return model.user_datastore[user, data_store] == path_component

    def disallow_access_to_other_pii_data_types(user, data_type):
        user_data_store_data_type_predicate = lambda user, data_store, data_type: \
            model.sum([model.user_datastore[user, data_store],
                       model.data.data_store_contains_data_type(data_store, data_type)]) == 2

        user_data_store_dynamic_data_type_predicate = lambda user, data_store, data_type: \
            model.sum([model.data.get_dynamic_user_datastore_accesses(user, data_store),
                       model.data.data_store_contains_data_type(data_store, data_type)]) == 2

        solution_accessed_data_types = lambda user, data_type: \
            model.sum([user_data_store_data_type_predicate(user, data_store, data_type) for data_store in
                       model.data.get_data_stores()]) >= 1

        dynamically_accessed_data_types = lambda user, data_type: \
            model.sum([user_data_store_dynamic_data_type_predicate(user, data_store, data_type) for data_store in
                       model.data.get_data_stores()]) >= 1

        return solution_accessed_data_types(user, data_type) <= dynamically_accessed_data_types(user, data_type)

    # constraint: allow_dynamic_accesses (Table 1, eq. 6)
    print("Adding allow_dynamic_accesses constraints")
    start_time = time.time()
    model.add([allow_dynamic_access(user, data_store) for user in tqdm(model.data.get_users()) for data_store in
               model.data.get_data_stores()])
    print("Execution time: allow_dynamic_accesses: {} sec".format(time.time() - start_time))

    # constraint: compute_datastore_access_groups # constraint: allow_dynamic_accesses (Table 1, eq. 7)
    print("Adding compute_datastore_access_groups constraints")
    start_time = time.time()
    model.add(
        [compute_datastore_access_groups(user, data_store) for user in tqdm(model.data.get_users()) for data_store in
         model.data.get_data_stores()])
    print("Execution time: compute_datastore_access_groups): {} sec".format(time.time() - start_time))

    # constraint: disallow_access_to_other_pii_data_types # constraint: allow_dynamic_accesses (Table 1, eq. 8)
    print("Adding disallow_access_to_other_pii_data_types constraints")
    start_time = time.time()
    model.add(
        [disallow_access_to_other_pii_data_types(user, data_type) for user in tqdm(model.data.get_users()) for data_type
         in
         model.data.get_data_types()])
    print("Execution time: disallow_access_to_other_pii_data_types): {} sec".format(time.time() - start_time))

    if monolithic_cp_exp:
        # add hard user dissimilarity constraints, which often lead to infeasible optimization problem,
        # thus, they are replaced with iteratively added user-dissimilarity constraints
        if kwargs["add_user_embedding_constraints"]:
            dissimilarity_user_data = list()
            users = model.data.get_users()

            for i in range(len(users)):
                user_a = users[i]

                for j in range(i + 1, len(users)):
                    user_b = users[j]
                    dissimilarity = model.data.get_users_dissimilarity(user_a, user_b)
                    if dissimilarity > kwargs["dissimilarity_threshold"]:
                        dissimilarity_user_data.append([(user_a, user_b), dissimilarity])

            print("Adding {} hard user dissimilarity constraints".format(len(dissimilarity_user_data)))
            add_user_xor_constraint(model, dissimilarity_user_data)

        # add hard datastore dissimilarity constraints, which often lead to infeasible optimization problem,
        # thus, they can be replaced with iteratively added datastore-dissimilarity constraints,
        # however, they are not used in the paper
        if kwargs["add_data_store_embedding_constraints"]:
            dissimilarity_data_store_data = list()
            data_stores = model.data.get_data_stores()

            for i in range(len(data_stores)):
                data_store_a = data_stores[i]

                for j in range(i + 1, len(data_stores)):
                    data_store_b = data_stores[j]
                    dissimilarity = model.data.get_data_stores_dissimilarity(data_store_a, data_store_b)
                    if dissimilarity > kwargs["dissimilarity_threshold"]:
                        dissimilarity_data_store_data.append([(data_store_a, data_store_b), dissimilarity])

            print("Adding {} hard data_store dissimilarity constraints".format(len(dissimilarity_data_store_data)))
            add_data_store_xor_constraint(model, dissimilarity_data_store_data)


def add_objective(model, **kwargs):
    """
    Add the model objective function (table 1, eq. 5)
    """

    def dark_permissions_total_count():
        """
        A component of the objective function that calculates the amount of dark permissions (Section 3, eq 2)
        """
        dark_permissions_per_user = lambda user: model.sum(
            [model.data.get_user_datastore_permissions(user, data_store) - model.user_datastore[user, data_store] for
             data_store in model.data.get_data_stores()])
        return -model.sum([dark_permissions_per_user(user) for user in model.data.get_users()])

    def dark_permissions_per_user_slack_variable(user):
        """
        Soft constraints that calculate the amount of dark on a per-user basis (Table 1, eq. 9, 10)
        """
        assigned_permissions = model.sum(
            [model.user_datastore[user, data_store] for data_store in model.data.get_data_stores()])
        required_permissions = model.sum([
            model.data.get_dynamic_user_datastore_accesses(user, data_store) for data_store in
            model.data.get_data_stores()])
        # Hinge loss
        loss = lambda x: model.max(0, x)
        x = assigned_permissions - (1 + kwargs["relative_dark_permission_budget"]) * required_permissions
        return loss(x)

    def limit_max_dark_permissions_per_user():
        return model.sum([dark_permissions_per_user_slack_variable(user) for user in model.data.get_users()])

    # Add maximization objective
    model.minimize(dark_permissions_total_count() + limit_max_dark_permissions_per_user())


def find_max_n_distinct_entities(entity_list, threshold, dissimilarity_func, max_constraints_per_group=None):
    """
    Return most dissimilar entities (e.g. users, datastores).

    Dissimilarity is defined by the dissimilarity_func function.  max_constraints_per_group parameter (if provided)
    controls the number of returned dissimilar entities per data access group. To consider dissimilar, the dissimilarity
    score between two entities should be above the threshold.
    """
    entity_similarity_submatrix = dict()

    for i in range(len(entity_list)):
        entity_a = entity_list[i]

        for j in range(i + 1, len(entity_list)):
            entity_b = entity_list[j]
            dissimilarity = dissimilarity_func(entity_a, entity_b)

            if dissimilarity >= threshold:
                entity_similarity_submatrix[(entity_a, entity_b)] = dissimilarity

    entity_similarity_submatrix = sorted(entity_similarity_submatrix.items(), key=lambda x: x[1], reverse=True)

    if max_constraints_per_group is not None:
        entity_similarity_submatrix = entity_similarity_submatrix[: max_constraints_per_group]

    return entity_similarity_submatrix


def add_user_xor_constraint(model, dissimilar_users: list) -> None:
    """
    Add hard constraints that disallow dissimilar users to share a data access group (Section 3)
    """

    def user_inverse_and_constraint(user_a, user_b, group):
        return model.sum(
            [model.user_data_access_group[(user_a, group)], model.user_data_access_group[(user_b, group)]]) <= 1

    for group in model.data.get_groups():
        for (user_a, user_b), dissimilarity in dissimilar_users:
            model.add(user_inverse_and_constraint(user_a, user_b, group))


def add_data_store_xor_constraint(model, dissimilar_data_stores: list) -> None:
    """
    Add hard constraints that disallow dissimilar datastores to share a data access group
    (not used in the current version of the paper)
    """

    def data_store_inverse_and_constraint(data_store_a, data_store_b, group):
        return model.sum([model.data_access_group_datastore[(group, data_store_a)],
                          model.data_access_group_datastore[(group, data_store_b)]]) <= 1

    for group in model.data.get_groups():
        for (data_store_a, data_store_b), dissimilarity in dissimilar_data_stores:
            model.add(data_store_inverse_and_constraint(data_store_a, data_store_b, group))


def compute_avg_entity_dissimilarity(entity_groups: dict, dissimilarity_func) -> float:
    """
    Compute the average dissimilarity of entities of a certain type - either users or datastores.
    """
    dissimilarity_values = list()

    for group, entities in entity_groups.items():
        for entity_a in entities:
            for entity_b in entities:
                val = dissimilarity_func(entity_a, entity_b)
                dissimilarity_values.append(val)

    return statistics.mean(dissimilarity_values) / 2


def compute_number_of_unsatisfied_constraints(entity_groups: dict, dissimilarity_func,
                                              threshold: float) -> float:
    """
    Returns the number of unsatisfied entity (use or datastore) dissimilarity constraints.
    """
    ctr = 0

    for group in entity_groups.keys():
        dissimilar_users = find_max_n_distinct_entities(entity_groups[group],
                                                        threshold=threshold,
                                                        dissimilarity_func=dissimilarity_func,
                                                        max_constraints_per_group=None)
        ctr += len(dissimilar_users)

    return ctr


def get_solution_user_data_store_edges(solution, all_var_solutions=None):
    """
    Extract $\widetilda{\\ud} from the solution (Section 3, "Computing $\widetilde{\\ud}(u, d)$")
    """
    reachable_data_stores_by_users = set()
    assert (solution is None) != (all_var_solutions is None)

    if solution is not None:
        if solution.get_solve_status() not in {"Optimal", "Feasible"}:
            return reachable_data_stores_by_users

        all_var_solutions = solution.get_all_var_solutions()

    for variable in all_var_solutions:
        var_name = variable.get_name()
        var_value = variable.get_value()

        if "user" in var_name and "ds" in var_name and var_value != 0:
            user, data_store = var_name.split("--")
            reachable_data_stores_by_users.add((user, data_store))

    return reachable_data_stores_by_users


def estimate_malicious_impact_after_optimization(solution_status, all_var_solutions, model_data, n):
    """
   Evaluate the impact of the worst-case attack after IAM optimization when n users get compromised
   """
    if solution_status not in {"Optimal", "Feasible"}:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "q1": None,
            "q3": None
        }

    users = model_data.get_users()
    data_stores = model_data.get_data_stores()

    def get_upper_bound():
        bound = 1

        for i in range(n):
            bound *= (len(users) - i)

        return bound

    data = list()
    max_iterations = min(100 * 1000, max(100 * 1000, get_upper_bound()))

    for iter in tqdm(range(max_iterations)):
        affected_data_stores = set()
        selected_users = random.sample(users, n)

        for user in selected_users:
            for data_store in data_stores:
                var_name = "{}--{}".format(user, data_store)
                var_value = 0

                if var_name in all_var_solutions:
                    var_value = all_var_solutions[var_name].value
                else:
                    print("ERROR: {} is not found in all_var_solutions".format(var_name))

                if var_value != 0:
                    affected_data_stores.add(data_store)

        data.append(len(affected_data_stores))

    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "q1": float(np.percentile(data, 25)),
        "q3": float(np.percentile(data, 75)),
    }


def estimate_malicious_impact_before_optimization(model_data, n):
    """
    Evaluate the impact of the worst-case attack before IAM optimization when n users get compromised
    """
    users = model_data.get_users()
    data_stores = model_data.get_data_stores()

    def get_upper_bound():
        bound = 1

        for i in range(n):
            bound *= (len(users) - i)

        return bound

    data = list()
    max_iterations = min(100 * 1000, max(100 * 1000, get_upper_bound()))

    for iter in tqdm(range(max_iterations)):
        affected_data_stores = set()
        selected_users = random.sample(users, n)

        for user in selected_users:
            for data_store in data_stores:
                permission = model_data.get_user_datastore_permissions(user, data_store)
                if permission != 0:
                    affected_data_stores.add(data_store)

        data.append(len(affected_data_stores))

    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "q1": float(np.percentile(data, 25)),
        "q3": float(np.percentile(data, 75)),
    }


def decode_solution(output_dir, solution):
    """
    De-anonymize the solution - only applicable to real-world instances.
    """
    if solution.get_solve_status() not in {"Optimal", "Feasible"}:
        return

    user_encoding = {v: k for k, v in solution.model.data.get_user_encoding().items()}
    data_stores_encoding = {v: k for k, v in solution.model.data.get_data_store_encoding().items()}

    user_groups = dict()
    group_data_stores = dict()

    for variable in solution.get_all_var_solutions():
        var_name = variable.get_name()
        var_value = variable.get_value()

        if "user" in var_name and "group" in var_name and var_value != 0:
            user, group = var_name.split("--")
            user = user_encoding[user]
            user_groups.setdefault(group, list())
            user_groups[group].append(user)

        if "group" in var_name and "ds" in var_name and var_value != 0:
            group, data_store = var_name.split("--")
            data_store = data_stores_encoding[data_store]
            group_data_stores.setdefault(group, list())
            group_data_stores[group].append(data_store)

    file_name = os.path.join(output_dir, "decoded_solution.json")
    with open(file_name, "w") as file:
        json.dump({
            "user_group_edges": user_groups,
            "group_data_store_edges": group_data_stores
        }, file)


def get_relative_dark_permissions_fraction(model, solution=None):
    """
    Compute the fraction of remaining dark permissions in the provided solution.

    If the solution is None, then returns 1.0, which means all dark permissions are still present in the model
    """
    if solution is None:
        return 1.0

    if solution.get_solve_status() not in {"Optimal", "Feasible"}:
        return None

    static_permission_count = 0
    dynamic_permission_count = 0
    permissions_given_by_solution_count = 0

    for user in model.data.get_users():
        for data_store in model.data.get_data_stores():
            static_permission_count += model.data.get_user_datastore_permissions(user, data_store)
            dynamic_permission_count += model.data.get_dynamic_user_datastore_accesses(user, data_store)

    for variable in solution.get_all_var_solutions():
        var_name = variable.get_name()
        var_value = variable.get_value()

        if "user" in var_name and "ds" in var_name and var_value != 0:
            permissions_given_by_solution_count += var_value

    return (permissions_given_by_solution_count - dynamic_permission_count) / (
            static_permission_count - dynamic_permission_count + 1E-6)


def get_max_group_diameter(dissimilarity_func, group_info):
    """
    Returns the max distance between any two entities (users, datastores) in a group.
    """
    max_group_diameter = float("-Inf")

    for group, entities in group_info.items():
        for entity_a in entities:
            for entity_b in entities:
                max_group_diameter = max(max_group_diameter, dissimilarity_func(entity_a, entity_b))

    return max_group_diameter


def solve(model, results, **kwargs):
    """
    Optimize the CP model, iteratively add user dissimilarity constraints, collect statistics
    """
    start_time = datetime.datetime.now()
    max_iterations = kwargs["max_iterations"]
    monolithic_cp_exp = kwargs["monolithic_cp_exp"]
    max_dissimilar_entity_pairs_per_group = kwargs["max_dissimilar_entity_pairs_per_group"]
    add_user_embedding_constraints = kwargs["add_user_embedding_constraints"]
    add_data_store_embedding_constraints = kwargs["add_data_store_embedding_constraints"]
    model.parameters = CpoParameters(TimeLimit=kwargs["time_limit_per_run"], TimeMode="ElapsedTime",
                                     RelativeOptimalityTolerance=kwargs["tolerance"])  # 1 hour per iteration at most
    dissimilarity_threshold = kwargs["dissimilarity_threshold"]
    solution = None
    stats = {
        "user_count": len(model.data.get_users()),
        "datastore_count": len(model.data.get_data_stores()),
        "group_count": kwargs["group_count"],
        "data_types_count": len(model.data.get_data_types()),
        "relative_dark_permissions_fraction": get_relative_dark_permissions_fraction(model),
        "max_user_group_diameter": get_max_group_diameter(model.data.get_users_dissimilarity,
                                                          group_info={"": model.data.get_users()}),
        "max_data_store_group_diameter": get_max_group_diameter(model.data.get_data_stores_dissimilarity,
                                                                group_info={"": model.data.get_data_stores()}),
        "last_feasible_solution": None,
        "solution_stats": list()
    }

    # iteratively add constraints (Section 3)
    for i in tqdm(range(max_iterations)):
        print("Iteration {}".format(i))
        solution = model.solve(log_output=True, TimeLimit=kwargs["time_limit_per_run"], solve_with_search_next=True,
                               execfile=config.CPO_OPTIMIZER_PATH)

        objective_value = solution.get_objective_value()
        relative_dark_permissions_fraction = get_relative_dark_permissions_fraction(model, solution)
        local_stats = {
            "iteration": i,
            "solve_status": solution.get_solve_status(),
            "objective": objective_value,
            "objective_bound": solution.get_objective_bounds()[0],
            "total_solve_time": solution.get_solve_time(),
            "variable_count": solution.solver_infos.get_number_of_integer_vars(),
            "constraints_count": solution.solver_infos.get_number_of_constraints(),
            "memory_usage": solution.solver_infos.get_memory_usage() / (1024 * 1024 * 1024),  # GB
            "relative_dark_permissions_fraction": relative_dark_permissions_fraction,
            "process_infos": solution.get_solver_infos()
        }
        stats["solution_stats"].append(local_stats)

        if solution.get_solve_status() not in {"Optimal", "Feasible"}:
            print("Solution is not found at iteration {}. Status: {} ".format(i, solution.get_solve_status()))
            return solution, stats

        stats["last_feasible_solution"] = solution
        local_stats["vars"] = solution.get_all_var_solutions()

        if i > 0:
            # if the dark_permissions_fraction quadruples, then stop
            prev_relative_dark_permissions_fraction = stats["solution_stats"][-2]["relative_dark_permissions_fraction"]
            relative_dark_permissions_upper_bound = max(0.2, 4 * prev_relative_dark_permissions_fraction)
            if relative_dark_permissions_fraction > relative_dark_permissions_upper_bound:
                print(
                    "Integrating embedding constraints has worsened the existing solution. dark_permissions: {} vs {}".format(
                        relative_dark_permissions_fraction, prev_relative_dark_permissions_fraction))
                return solution, stats

        user_groups = dict()
        data_store_groups = dict()

        for variable in solution.get_all_var_solutions():
            var_name = variable.get_name()
            var_value = variable.get_value()

            if "user" in var_name and "group" in var_name and var_value != 0:
                user, group = var_name.split("--")
                user_groups.setdefault(group, list())
                user_groups[group].append(user)

            if "ds" in var_name and "group" in var_name and var_value != 0:
                group, ds = var_name.split("--")
                data_store_groups.setdefault(group, list())
                data_store_groups[group].append(ds)

        local_stats["user_dissimilarity"] = compute_avg_entity_dissimilarity(user_groups,
                                                                             model.data.get_users_dissimilarity)
        local_stats["data_store_dissimilarity"] = compute_avg_entity_dissimilarity(data_store_groups,
                                                                                   model.data.get_data_stores_dissimilarity)
        local_stats["unsatisfied_user_constraints_count"] = compute_number_of_unsatisfied_constraints(user_groups,
                                                                                                      model.data.get_users_dissimilarity,
                                                                                                      dissimilarity_threshold)
        local_stats["unsatisfied_data_store_constraints_count"] = compute_number_of_unsatisfied_constraints(
            data_store_groups, model.data.get_data_stores_dissimilarity, dissimilarity_threshold)

        local_stats["max_user_group_diameter"] = get_max_group_diameter(model.data.get_users_dissimilarity,
                                                                        group_info=user_groups)
        local_stats["max_data_store_group_diameter"] = get_max_group_diameter(model.data.get_data_stores_dissimilarity,
                                                                              group_info=data_store_groups)

        # check if iterative constraint are enabled
        if monolithic_cp_exp:
            break

        for group in model.data.get_groups():
            # add user dissimilarity constraints (Section 3.2)
            if add_user_embedding_constraints and group in user_groups:
                dissimilar_users = find_max_n_distinct_entities(user_groups[group],
                                                                threshold=dissimilarity_threshold,
                                                                dissimilarity_func=model.data.get_users_dissimilarity,
                                                                max_constraints_per_group=max_dissimilar_entity_pairs_per_group)
                add_user_xor_constraint(model, dissimilar_users)
                local_stats.setdefault("added_user_xor_constraints_count", 0)
                local_stats["added_user_xor_constraints_count"] += len(dissimilar_users)

            # add datastore dissimilarity constraints (Section 3.2)
            if add_data_store_embedding_constraints and group in data_store_groups:
                dissimilar_data_stores = find_max_n_distinct_entities(data_store_groups[group],
                                                                      threshold=dissimilarity_threshold,
                                                                      dissimilarity_func=model.data.get_data_stores_dissimilarity,
                                                                      max_constraints_per_group=max_dissimilar_entity_pairs_per_group)
                add_data_store_xor_constraint(model, dissimilar_data_stores)
                local_stats.setdefault("added_data_store_xor_constraints_count", 0)
                local_stats["added_data_store_xor_constraints_count"] += len(dissimilar_data_stores)

        model.set_starting_point(solution.solution)
        prev_user_dissimilarity_value = stats["solution_stats"][-2]["user_dissimilarity"] if len(
            stats["solution_stats"]) > 1 else float("-inf")
        user_dissimilarity_value_change = prev_user_dissimilarity_value - local_stats["user_dissimilarity"]

        print("\nAvg. user dissimilarity {} at iteration {}".format(local_stats["user_dissimilarity"], i))
        print("Avg. data_store dissimilarity {} at iteration {}".format(local_stats["data_store_dissimilarity"], i))
        print("# of unsatisfied USER constraints {} and DATA_STORE constraints {} at iteration {}".format(
            local_stats["unsatisfied_user_constraints_count"], local_stats["unsatisfied_data_store_constraints_count"],
            i))
        print("# of added unsatisfied USER constraints {} and DATA_STORE constraints {} at iteration {}".format(
            local_stats.get("added_user_xor_constraints_count", None),
            local_stats.get("added_data_store_xor_constraints_count", None), i))
        print("objective value {} at iteration {}".format(local_stats["objective"], i))

        if add_user_embedding_constraints and add_data_store_embedding_constraints:
            if local_stats["unsatisfied_user_constraints_count"] + local_stats[
                "unsatisfied_data_store_constraints_count"] <= 0:
                print("Done: no more unsatisfied constraints")
                break
        elif add_user_embedding_constraints:
            if local_stats["unsatisfied_user_constraints_count"] <= 0:
                print("Done: no more unsatisfied user constraints")
                break
        elif add_data_store_embedding_constraints:
            if local_stats["unsatisfied_data_store_constraints_count"] <= 0:
                print("Done: no more unsatisfied data_store constraints")
                break

        if start_time + datetime.timedelta(seconds=kwargs["total_time_limit"]) < datetime.datetime.now():
            print("{}-second time out".format(kwargs["total_time_limit"]))
            break

        last_feasible_solution = stats.pop("last_feasible_solution", None)
        stats["org"] = kwargs["org"]
        results[str(kwargs["group_count"])] = stats
        joblib.dump(results, kwargs["output_file_name"])
        stats["last_feasible_solution"] = last_feasible_solution

    return solution, stats


def build(*args, **kwargs) -> docplex.cp.model.CpoModel:
    """
    Build CpoModel object
    """
    model = CpoModel(name="iam")
    model.data = kwargs["data"]
    add_variables(model, **kwargs)
    add_constraints(model, **kwargs)
    add_objective(model, **kwargs)
    model.add_solver_listener(
        MonitorRelativeDarkPermissionReduction(dark_permission_tolerance=kwargs["dark_permission_tolerance"]))
    model.print_information()
    return model


def get_max_k_impact(k, sparse_adj_matrix: set):
    """
    Evaluate the impact the worst-case attack when k users get compromised
    """
    compromised_data_stores = set()
    selected_users = set()

    if len(sparse_adj_matrix) == 0:
        return selected_users, compromised_data_stores

    for i in range(k):
        user_ds_additional_coverage = dict()

        for (user, ds) in sparse_adj_matrix:
            if user in selected_users or ds in compromised_data_stores:
                continue

            user_ds_additional_coverage.setdefault(user, list())
            user_ds_additional_coverage[user].append(ds)

        if len(user_ds_additional_coverage) == 0:
            break

        sorted_data = sorted(user_ds_additional_coverage.items(), key=lambda x: len(x[1]), reverse=True)
        user, data_stores = sorted_data[0]
        selected_users.add(user)
        compromised_data_stores.update(data_stores)

    return selected_users, compromised_data_stores


def get_max_k_impact_wrt_clsuters(k, user_clusters, sparse_adj_matrix: set):
    """
    Evaluate the impact the worst-case attack when k users get compromised and clusters are taken into consideration
    """
    compromised_data_stores = dict()
    selected_users = dict()

    if len(sparse_adj_matrix) == 0:
        return set(), set(), -1

    for cluster, cluster_users in user_clusters.items():
        cluster_users = set(cluster_users)
        compromised_data_stores.setdefault(cluster, set())
        selected_users.setdefault(cluster, set())

        for i in range(k):
            user_ds_additional_coverage = dict()

            for (user, ds) in sparse_adj_matrix:
                if user in selected_users[cluster] or ds in compromised_data_stores[cluster]:
                    continue

                if user not in cluster_users:
                    continue

                user_ds_additional_coverage.setdefault(user, list())
                user_ds_additional_coverage[user].append(ds)

            if len(user_ds_additional_coverage) == 0:
                break

            sorted_data = sorted(user_ds_additional_coverage.items(), key=lambda x: len(x[1]), reverse=True)
            user, data_stores = sorted_data[0]
            selected_users[cluster].add(user)
            compromised_data_stores[cluster].update(data_stores)

    compromised_data_stores_sorted_list = sorted(compromised_data_stores.items(), key=lambda x: len(x[1]), reverse=True)
    cluster, compromised_data_stores = compromised_data_stores_sorted_list[0]
    selected_users_list = selected_users[cluster]
    return selected_users_list, compromised_data_stores, cluster


def extract_feasible_solution(solution, stats):
    """
    Extract feasible solution
    """
    stats["last_solution_stats"] = stats["solution_stats"][-1]

    if solution.get_solve_status() not in {"Optimal", "Feasible"}:
        if stats["last_feasible_solution"] is not None:
            print("Solution not found. Replacing with the last feasible solution")
            solution = stats["last_feasible_solution"]
            stats["last_solution_stats"] = stats["solution_stats"][-2]

    return solution, stats


def compute_attack_impact_stats(solution):
    """
    Evaluate the impact of IAMAX's policy optimization on simulated attacks.
    """
    max_compromised_users = 3
    max_attack_stats = {"before": dict(), "after": dict()}
    avg_attack_stats = {"before": dict(), "after": dict()}

    for user_count in range(1, max_compromised_users):
        # max impact before an optimization
        original_selected_users, original_compromised_data_stores = get_max_k_impact(k=max_compromised_users,
                                                                                     sparse_adj_matrix=solution.model.data.identity_datastore_permissions_set)
        max_attack_stats["before"][str(user_count)] = len(original_compromised_data_stores)
        # average impact before optimization
        avg_attack_stats["before"][str(user_count)] = estimate_malicious_impact_before_optimization(solution.model.data,
                                                                                                    user_count)

        # max impact after an optimization
        selected_users, compromised_data_stores = get_max_k_impact(k=max_compromised_users,
                                                                   sparse_adj_matrix=get_solution_user_data_store_edges(
                                                                       solution))
        max_attack_stats["after"][str(user_count)] = len(compromised_data_stores)
        # average impact after optimization
        all_var_solutions_dict = {x.get_name(): x for x in solution.get_all_var_solutions()}
        avg_attack_stats["after"][str(user_count)] = estimate_malicious_impact_after_optimization(
            solution.get_solve_status(), all_var_solutions_dict, solution.model.data, user_count)

    print("Avg impact BEFORE the optimization:")
    for key, value in avg_attack_stats["before"].items():
        print(key, value)

    print("\nMax impact BEFORE the optimization:\n{}".format(max_attack_stats["before"]))

    print("\nAvg impact AFTER the optimization:")
    for key, value in avg_attack_stats["after"].items():
        print(key, value)

    print("\nMax impact AFTER the optimization:\n{}".format(max_attack_stats["after"]))

    return {
        "max_attack_stats": max_attack_stats,
        "avg_attack_stats": avg_attack_stats
    }


def run_experiments(data, parameters, results):
    """
    Run experiment that are reported in the IAMAX paper
    """
    org_dir = parameters["org_dir"]
    # choose the most dissimilar 1% of users to spread them across different groups
    dissimilarity_threshold = parameters["dissimilarity_threshold"]
    print("embeddings_dissimilarity_threshold", dissimilarity_threshold)

    # build CP model
    parameters["data"] = data
    model = build(**parameters)
    solution, stats = solve(model, results, **parameters)
    solution, stats = extract_feasible_solution(solution, stats)

    # Save the de-anonymized CP solution to the file "solution.json"
    output_dir = os.path.join(org_dir, "output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.mkdir(output_dir)
    decode_solution(output_dir, solution)
    return solution, stats
