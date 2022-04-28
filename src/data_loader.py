#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Mikhail Kazdagli"
__copyright__ = "Copyright 2021, Symmetry Systems Inc"

import json

import joblib
import numpy as np
import pandas as pd
from docplex.mp.model import *


class Data:
    """
    Loads data from a disk and prepares for being used within a CP program.

    Each dataset is stored in a separate folder and it contains multiple files.
    """

    def __init__(self, org_dir, global_parameters, all_var_solutions=None, log_level=1):
        self.log_level = log_level
        self.global_parameters = global_parameters
        self.org_dir = org_dir
        self.load_users()
        self.load_data_stores()
        self.load_data_types()
        self.load_user_datastore_permissions()
        self.load_dynamic_user_datastore_accesses()
        self.load_datastore_data_type()
        self.load_users_dissimilarity()
        self.load_data_stores_dissimilarity()
        self.load_user_clusters()
        self.fix_data_inconsistencies()
        self.keep_active_entities(all_var_solutions)
        self.compute_max_cluster_diameter()

    def load_users(self):
        """
        Load users.
        """
        file_name = os.path.join(self.org_dir, "user_encoding.joblib")
        self.user_encoding = joblib.load(file_name)
        self.users = list(self.user_encoding.values())

    def load_data_stores(self):
        """
        Load datastores.
        """
        file_name = os.path.join(self.org_dir, "data_stores_encoding.joblib")
        self.data_stores_encoding = joblib.load(file_name)
        self.data_stores = list(self.data_stores_encoding.values())

    def load_data_types(self):
        """
        Load data types.
        """
        file_name = os.path.join(self.org_dir, "datastore_pii.json")

        with open(file_name, "r") as file:
            data = json.load(file)

        self.data_types = sorted({x for _, pii_types in data.items() for x in pii_types})

    # load only active entries
    def load_user_datastore_permissions(self):
        """
        Load user-datastores permissions (Section 3, UD variable)
        """
        file_name = os.path.join(self.org_dir, "user_data_store_permissions_set.joblib")

        if not os.path.exists(file_name):
            file_name = os.path.join(self.org_dir, "user_datastore_permissions_dict.joblib")

        data = joblib.load(file_name)

        if isinstance(data, set):
            self.user_datastore_permissions_set = data
        else:
            # convert into integers
            data = {k: int(v) for k, v in data.items()}
            self.user_datastore_permissions_set = {k for k, v in data.items() if v}

        print("Loaded {} permission edges".format(len(self.user_datastore_permissions_set)))

    def load_dynamic_user_datastore_accesses(self):
        """
        Load user-datastores dynamic accesses (Section 3, $\widetilde{\\ud}(u, d)$ variable)
        """
        file_name = os.path.join(self.org_dir, "user_data_store_dynamic_accesses_set.joblib")

        if not os.path.exists(file_name):
            file_name = os.path.join(self.org_dir, "user_data_store_dynamic_accesses_dict.joblib")

        data = joblib.load(file_name)

        if isinstance(data, set):
            self.dynamic_user_datastore_accesses_set = data
        else:
            # convert into integers
            data = {k: int(v) for k, v in data.items()}
            self.dynamic_user_datastore_accesses_set = {k for k, v in data.items() if v}

        print("Loaded {} dynamic edges".format(len(self.dynamic_user_datastore_accesses_set)))

    def load_datastore_data_type(self):
        """
        Load information about data types (Section 3, DT variable).
        """
        file_name = os.path.join(self.org_dir, "data_store_data_type_set.joblib")

        if not os.path.exists(file_name):
            file_name = os.path.join(self.org_dir, "datastore_data_type_dict.joblib")

        data = joblib.load(file_name)

        if isinstance(data, set):
            self.data_store_data_type_set = data
        else:
            data = {k: int(v) for k, v in data.items()}
            self.data_store_data_type_set = {k: int(v) for k, v in data.items() if v}

    def load_users_dissimilarity(self):
        """
        Load user dissimilarity matrix pre-computed using GNN (Section 3.2)
        """
        file_name = os.path.join(self.org_dir, "users_dissimilarity_dict.joblib")
        self.users_dissimilarity_dict = joblib.load(file_name)

    def load_data_stores_dissimilarity(self):
        """
        Load datastore dissimilarity matrix pre-computed using GNN (not being used)
        """
        self.data_store_dissimilarity_dict = dict()
        file_name = os.path.join(self.org_dir, "data_store_dissimilarity_dict.joblib")

        if os.path.exists(file_name):
            self.data_store_dissimilarity_dict = joblib.load(file_name)

    def load_user_clusters(self):
        """
        Load pre-computed cluster assignment for users (Section 4)
        """
        self.user_clusters = dict()
        file_name = os.path.join(self.org_dir, "source_identity_clusters.parquet")

        # we do not use cluster information with synthetic instances
        if not os.path.exists(file_name) and "synthetic" in self.org_dir:
            self.user_clusters["0"] = list(self.get_users())
            return

        df = pd.read_parquet(file_name)

        for user, cluster_id in df["cluster_id"].to_dict().items():
            self.user_clusters.setdefault(cluster_id, list())
            self.user_clusters[cluster_id].append(user)

    def get_cluster_id(self, user):
        """
        Get cluster ID for a specific user.
        """
        for cluster_id, users in self.user_clusters.items():
            if user in users:
                return cluster_id

        return None

    def get_active_users(self):
        """
        Get active users - the users that access any data.
        """
        active_users = {user for user, data_store in self.dynamic_user_datastore_accesses_set}
        return active_users

    def get_active_data_stores(self, active_users):
        """
        Get active datastores - datastores that being accessed.
        """
        active_data_stores = {data_store for user, data_store in self.dynamic_user_datastore_accesses_set if
                              user in active_users}
        return active_data_stores

    def fix_data_inconsistencies(self):
        """
        Fix data inconsistencies encountered in real data sets.
        """
        error_ctr = 0

        for user, ds in self.dynamic_user_datastore_accesses_set:
            if (user, ds) not in self.user_datastore_permissions_set:
                error_ctr += 1
                if self.log_level > 0:
                    print("Data inconsistency #{}: user {} does not have a permission edge to the data store {}".format(
                        error_ctr, user, ds))
                self.user_datastore_permissions_set.add((user, ds))

    def keep_active_entities(self, all_var_solutions):
        """
        Remove inactive users from other data structures.
        """
        if all_var_solutions is None:
            active_users = self.get_active_users()
            active_data_stores = self.get_active_data_stores(active_users)
        else:
            active_users_ = {x.get_name().split("--")[0] for x in all_var_solutions}
            active_users = {x for x in active_users_ if "user" in x}

            active_data_stores_ = {x.get_name().split("--")[1] for x in all_var_solutions}
            active_data_stores = {x for x in active_data_stores_ if "ds" in x}

        print("Loaded {} active users out of {}".format(len(active_users), len(self.users)))
        print("Loaded {} active data stores out of {}".format(len(active_data_stores), len(self.data_stores)))

        self.user_datastore_permissions_set = {(user, ds) for user, ds in self.user_datastore_permissions_set
                                               if user in active_users and ds in active_data_stores}

        self.dynamic_user_datastore_accesses_set = {(user, ds) for user, ds in
                                                    self.dynamic_user_datastore_accesses_set
                                                    if user in active_users and ds in active_data_stores}

        self.data_store_data_type_set = {(ds, data_type) for ds, data_type in self.data_store_data_type_set if
                                         ds in active_data_stores}

        self.users_dissimilarity = {(user_a, user_b): val for (user_a, user_b), val in
                                    self.users_dissimilarity_dict.items()
                                    if user_a in active_users and user_b in active_users}

        self.data_stores_dissimilarity = {(ds_a, ds_b): val for (ds_a, ds_b), val in
                                          self.data_store_dissimilarity_dict.items()
                                          if ds_a in active_data_stores and ds_b in active_data_stores}
        self.users = list(active_users)
        self.data_stores = list(active_data_stores)

        active_user_clusters = dict()
        for cluster_id, users in self.user_clusters.items():
            for user in users:
                if user in active_users:
                    active_user_clusters.setdefault(cluster_id, list())
                    active_user_clusters[cluster_id].append(user)
        self.user_clusters = active_user_clusters

    def compute_max_cluster_diameter(self):
        """
        Compute a cluster's diameter.
        """
        self.max_user_cluster_diameter = float("-Inf")

        for cluster_id, users in self.user_clusters.items():
            for user_a in users:
                for user_b in users:
                    dissimilarity = self.get_users_dissimilarity(user_a, user_b)
                    self.max_user_cluster_diameter = max(self.max_user_cluster_diameter, dissimilarity)

    def set_group_count(self, group_count):
        """
        Update the number of groups property stored in this class.
        """
        self.groups = ["group_" + str(i) for i in range(group_count)]

    def get_users(self):
        return self.users

    def get_groups(self):
        return self.groups

    def get_data_stores(self):
        return self.data_stores

    def get_data_types(self):
        return self.data_types

    def get_user_encoding(self):
        return self.user_encoding

    def get_data_store_encoding(self):
        return self.data_stores_encoding

    def get_users_dissimilarity(self, user_a, user_b):
        """
        Get a pre-computed dissimilarity score between two users.
        """
        if not self.global_parameters["add_user_embedding_constraints"]:
            return 0

        if user_a == user_b:
            return 0

        if (user_a, user_b) in self.users_dissimilarity:
            return self.users_dissimilarity[user_a, user_b]

        return self.users_dissimilarity[user_b, user_a]

    def get_data_stores_dissimilarity(self, data_store_a, data_store_b):
        """
        Get a pre-computed dissimilarity score between two datastores.
        """
        if not self.global_parameters["add_data_store_embedding_constraints"]:
            return 0

        if data_store_a == data_store_b:
            return 0

        if (data_store_a, data_store_b) in self.data_stores_dissimilarity:
            return self.data_stores_dissimilarity[data_store_a, data_store_b]

        return self.data_stores_dissimilarity[data_store_b, data_store_a]

    def get_user_clusters(self):
        return self.user_clusters

    def get_dynamic_user_datastore_accesses(self, user, data_store):
        """
        Check if the given user accesses the provided datastore.
        """
        if (user, data_store) in self.dynamic_user_datastore_accesses_set:
            return 1

        return 0

    def get_user_datastore_permissions(self, user, data_store):
        """
        Check if the given user has a permission to access the provided datastore.
        """

        # in case the permission model is missing some dynamic accesses
        if self.get_dynamic_user_datastore_accesses(user, data_store) == 1:
            return 1

        if (user, data_store) in self.user_datastore_permissions_set:
            return 1

        return 0

    def data_store_contains_data_type(self, data_store, data_type):
        """
        Check if the given datastore contains the requested data type.
        """
        if (data_store, data_type) in self.data_store_data_type_set:
            return 1

        return 0

    def get_user_dissimilarity_threshold(self, global_parameters):
        """
        Compute the dissimilarity threshold above which two users are considered dissimilar.
        Only active users are considered during this computation.
        """
        add_user_embedding_constraints = global_parameters["add_user_embedding_constraints"]
        add_data_store_embedding_constraints = global_parameters["add_data_store_embedding_constraints"]
        percentile = global_parameters["percentile"]
        user_objects = dict()
        data_store_objects = dict()

        if add_user_embedding_constraints:
            for i in range(len(self.users)):
                user_a = self.users[i]
                for j in range(i + 1, len(self.users)):
                    user_b = self.users[j]
                    val = self.get_users_dissimilarity(user_a, user_b)
                    user_objects[user_a, user_b] = val

        if add_data_store_embedding_constraints:
            for i in range(len(self.data_stores)):
                data_store_a = self.data_stores[i]
                for j in range(len(self.data_stores)):
                    data_store_b = self.data_stores[j]
                    val = self.get_data_stores_dissimilarity(data_store_a, data_store_b)
                    data_store_objects[data_store_a, data_store_b] = val

        prcn = np.percentile(list(user_objects.values()) + list(data_store_objects.values()), percentile)
        return prcn

    def get_violated_constraints(self, global_parameters, threshold):
        add_user_embedding_constraints = global_parameters["add_user_embedding_constraints"]
        add_data_store_embedding_constraints = global_parameters["add_data_store_embedding_constraints"]
        user_objects = dict()
        data_store_objects = dict()

        if add_user_embedding_constraints:
            for i in range(len(self.users)):
                user_a = self.users[i]
                for j in range(i + 1, len(self.users)):
                    user_b = self.users[j]
                    val = self.get_users_dissimilarity(user_a, user_b)
                    user_objects[user_a, user_b] = val

        if add_data_store_embedding_constraints:
            for i in range(len(self.data_stores)):
                data_store_a = self.data_stores[i]
                for j in range(len(self.data_stores)):
                    data_store_b = self.data_stores[j]
                    val = self.get_data_stores_dissimilarity(data_store_a, data_store_b)
                    data_store_objects[data_store_a, data_store_b] = val

        violated_user_objects = [(entity_a, entity_b) for (entity_a, entity_b), val in user_objects.items() if
                                 val >= threshold]
        violated_data_store_objects = [(entity_a, entity_b) for (entity_a, entity_b), val in data_store_objects.items()
                                       if val >= threshold]
        return violated_user_objects, violated_data_store_objects
