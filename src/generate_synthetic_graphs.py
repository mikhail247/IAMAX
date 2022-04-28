#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "Mikhail Kazdagli"
__copyright__ = "Copyright 2021, Symmetry Systems Inc"

import json
import math
import os
import random

import joblib
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import pairwise_distances

import config
from src.data_loader import Data


class GenerateSyntheticGraphs:
    """
    The class used to generate synthetic datasets that statistically similar to real datasets.
    """
    metric = "cosine"

    def __init__(self, parameters, data):
        self.parameters = parameters
        self.org = parameters["org"]
        self.org_dir = os.path.join(config.DATA_DIR, self.org)
        self.init_real_graph_stats(data)

    def init_real_graph_stats(self, data):
        if not os.path.exists(self.org_dir):
            return

        nodes = len(data.get_users()) * len(data.get_data_stores())
        permission_edges = len(data.identity_datastore_permissions_set)
        dynamic_access_edges = len(data.dynamic_user_datastore_accesses_set)
        self.permission_graph_density = permission_edges / nodes
        self.dynamic_access_graph_density = dynamic_access_edges / nodes

    def load_object_as_joblib(self, file_name):
        file_name = os.path.join(self.org_dir, file_name)
        return joblib.load(file_name)

    @staticmethod
    def generate_random_embedding(vec):
        """
        Generate a random embedding vector if the corresponding embedding vector is not found.
        """
        if isinstance(vec, float) and math.isnan(vec):
            vec = np.random.normal(loc=0.0, scale=1.0, size=(config.EMBEDDING_SIZE,))

        return vec

    def sample_users(self, user_count, embedding_noise_std):
        """
        Sample user_count users from a real dataset and add Gaussian noise to their embeddings.
        """
        users = dict()
        user_encoding = self.load_object_as_joblib("user_encoding.joblib")
        user_pool = list(user_encoding.keys())
        sampled_users = np.random.choice(user_pool, size=user_count, replace=True)

        df_embeddings = pd.read_parquet(os.path.join(self.org_dir, "user_embeddings.parquet"))
        df_sampled_embeddings = df_embeddings.reindex(sampled_users)
        df_sampled_embeddings["embedding"] = df_sampled_embeddings["embedding"].apply(
            lambda vec: GenerateSyntheticGraphs.generate_random_embedding(vec))

        for i in range(user_count):
            row = df_sampled_embeddings.iloc[i, :]
            user_name = "{}_sample_id_{}".format(user_encoding[row.name], i)
            embedding = row["embedding"]
            embedding_noise = np.random.normal(loc=0.0, scale=embedding_noise_std, size=embedding.shape[0])
            embedding = embedding + embedding_noise
            embedding /= np.linalg.norm(embedding, ord=2)

            users[user_name] = {
                "user_name": user_name,
                "embedding": embedding
            }

        return users

    def sample_data_stores(self, data_store_count, embedding_noise_std):
        """
        Sample data_store_count datastores from a real dataset and add Gaussian noise to their embeddings.
        """
        data_stores = dict()
        # load PII data types
        file_name = os.path.join(self.org_dir, "datastore_pii.json")
        with open(file_name, "r") as file:
            data_store_pii = json.load(file)

        # load embeddings
        embedding_file_name = os.path.join(self.org_dir, "data_store_embeddings.parquet")
        data_store_encoding = self.load_object_as_joblib("data_stores_encoding.joblib")
        data_store_pool = list(data_store_encoding.keys())
        sampled_data_stores = np.random.choice(data_store_pool, size=data_store_count, replace=True)
        df_embeddings = pd.read_parquet(embedding_file_name)
        df_sampled_embeddings = df_embeddings.reindex(sampled_data_stores)
        df_sampled_embeddings["embedding"] = df_sampled_embeddings["embedding"].apply(
            lambda vec: GenerateSyntheticGraphs.generate_random_embedding(vec))

        for i in range(data_store_count):
            row = df_sampled_embeddings.iloc[i, :]
            data_store_name = "{}_sample_id_{}".format(data_store_encoding[row.name], i)
            embedding = row["embedding"]
            embedding_noise = np.random.normal(loc=0.0, scale=embedding_noise_std, size=embedding.shape[0])
            embedding = embedding + embedding_noise
            embedding /= np.linalg.norm(embedding, ord=2)

            data_stores[data_store_name] = {
                "data_store_name": data_store_name,
                "embedding": embedding,
                "pii": data_store_pii.get(row.name, list())
            }

        return data_stores

    def sample_edges(self, users, data_stores, permission_graph_density, dynamic_access_graph_density):
        """
        Generate random permission and dynamic edges such that generated graphs match real graphs wrt graph density.
        """

        def build_empty_adj_matrix():
            adj_matrix = {user_name: dict() for user_name in users.keys()}

            for key in adj_matrix.keys():
                adj_matrix[key] = {data_store_name: False for data_store_name in data_stores.keys()}

            return adj_matrix

        permission_edge_counter = 0
        dynamic_access_edge_counter = 0
        user_data_store_permissions_dict = build_empty_adj_matrix()
        user_data_store_dynamic_accesses_dict = build_empty_adj_matrix()

        permission_edge_count = math.ceil(permission_graph_density * len(users) * len(data_stores))
        dynamic_access_edge_count = math.ceil(dynamic_access_graph_density * len(users) * len(data_stores))
        assert dynamic_access_edge_count <= permission_edge_count

        users_list = list(users.values())
        data_store_names = [item["data_store_name"] for item in data_stores.values()]
        data_store_embeddings = [item["embedding"] for item in data_stores.values()]
        data_store_embeddings = np.vstack(data_store_embeddings)

        while permission_edge_counter < permission_edge_count:
            user = random.choice(users_list)
            user_embedding = user["embedding"]

            similarities = np.dot(data_store_embeddings, user_embedding)
            multinomial_distribution = scipy.special.softmax(similarities)
            data_store_index = np.argmax(np.random.multinomial(1, multinomial_distribution))
            data_store_name = data_store_names[data_store_index]

            if not user_data_store_permissions_dict[user["user_name"]][data_store_name]:
                user_data_store_permissions_dict[user["user_name"]][data_store_name] = True
                permission_edge_counter += 1

                if dynamic_access_edge_counter < dynamic_access_edge_count:
                    user_data_store_dynamic_accesses_dict[user["user_name"]][data_store_name] = True
                    dynamic_access_edge_counter += 1

        return user_data_store_permissions_dict, user_data_store_dynamic_accesses_dict

    @staticmethod
    def flatten_nested_dict(obj):
        new_obj = set()

        for key_outer in obj.keys():
            for key_inner, val in obj[key_outer].items():
                if val != 0:
                    new_obj.add((key_outer, key_inner))

        return new_obj

    def save(self, directory, users, data_stores, user_data_store_permissions_dict,
             user_data_store_dynamic_accesses_dict, parameters):
        """
        Save datastructures that constituent a dataset to separate files on disk.
        """
        file_name = os.path.join(directory, "parameters.joblib")
        joblib.dump(parameters, file_name)

        file_name = os.path.join(directory, "user_encoding.joblib")
        data = {key: key for key in users.keys()}
        joblib.dump(data, file_name)

        file_name = os.path.join(directory, "data_stores_encoding.joblib")
        data = {key: key for key in data_stores.keys()}
        joblib.dump(data, file_name)

        file_name = os.path.join(directory, "datastore_pii.json")
        data = {key: value["pii"] for key, value in data_stores.items()}
        with open(file_name, "w") as file:
            json.dump(data, file)

        file_name = os.path.join(directory, "user_datastore_permissions_dict.joblib")
        data = GenerateSyntheticGraphs.flatten_nested_dict(user_data_store_permissions_dict)
        joblib.dump(data, file_name)

        file_name = os.path.join(directory, "user_data_store_dynamic_accesses_dict.joblib")
        data = GenerateSyntheticGraphs.flatten_nested_dict(user_data_store_dynamic_accesses_dict)
        joblib.dump(data, file_name)

        pii_data_types = sorted({pii_type for value in data_stores.values() for pii_type in value["pii"]})
        datastore_data_type_dict = {(ds, pii_type): False for ds in data_stores.keys() for pii_type in pii_data_types}
        for ds in data_stores.values():
            for pii_type in pii_data_types:
                if pii_type in ds["pii"]:
                    datastore_data_type_dict[(ds["data_store_name"], pii_type)] = True
        file_name = os.path.join(directory, "datastore_data_type_dict.joblib")
        joblib.dump(datastore_data_type_dict, file_name)

        embeddings = np.vstack([item["embedding"] for item in users.values()])
        users_dissimilarity_matrix = pairwise_distances(embeddings, metric=GenerateSyntheticGraphs.metric)
        users_dissimilarity_dict = dict()
        for i, user_a_id in enumerate(users.keys()):
            for j, user_b_id in enumerate(users.keys()):
                users_dissimilarity_dict[(user_a_id, user_b_id)] = users_dissimilarity_matrix[i, j]
        file_name = os.path.join(directory, "users_dissimilarity_dict.joblib")
        joblib.dump(users_dissimilarity_dict, file_name)

    def generate_graph(self):
        """
        Generate a synthetic graph based on a specific real graph.
        """
        if not os.path.exists(self.org_dir):
            return

        synthetic_dir = os.path.join(config.DATA_DIR, "synthetic_graphs")
        output_directory = os.path.join(synthetic_dir, "{}_graph_{}".format(self.org, self.parameters["index"]))

        if os.path.exists(output_directory):
            return

        user_count = self.parameters["user_count"]
        data_store_count = self.parameters["data_store_count"]
        embedding_noise_std = self.parameters["embedding_noise_std"]
        permission_graph_density = self.permission_graph_density
        dynamic_access_graph_density = self.dynamic_access_graph_density
        users = self.sample_users(user_count, embedding_noise_std)
        data_stores = self.sample_data_stores(data_store_count, embedding_noise_std)
        user_data_store_permissions_dict, user_data_store_dynamic_accesses_dict = self.sample_edges(users, data_stores,
                                                                                                    permission_graph_density,
                                                                                                    dynamic_access_graph_density)

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        self.save(output_directory, users, data_stores, user_data_store_permissions_dict,
                  user_data_store_dynamic_accesses_dict, self.parameters)


def generate_data(graph_count=35):
    """
    Generate graph_count synthetic datasets for each real dataset.
    """
    global_parameters = {
        "add_user_embedding_constraints": False,
        "add_data_store_embedding_constraints": False,
        "monolithic_cp_exp": False
    }
    data_objects = dict()

    for org in config.ORG_ANONYMIZER.keys():
        print(org)
        if org not in data_objects:
            org_dir = os.path.join(config.DATA_DIR, org)
            data_objects[org] = Data(org_dir, global_parameters)

        for i in range(graph_count):
            parameters = dict(
                index=i,
                user_count=random.randint(10, 100),
                data_store_count=random.randint(50, 300),
                embedding_noise_std=1E-2,
                org=org
            )
            print("Processing organization: {} Synthetic graph_id: {}".format(org, i))
            print(parameters)
            obj = GenerateSyntheticGraphs(parameters, data_objects[org])
            obj.generate_graph()


if __name__ == "__main__":
    """
    Generate n synthetic datasets for each real dataset.
    """
    n = 35
    generate_data(graph_count=n)
