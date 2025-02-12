import csv
import umap
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from collections import defaultdict
import pyvis
from pyvis.network import Network
import numpy as np
import seaborn as sns
import branca.colormap as cm
import branca
import pandas as pd
import re
from textwrap import wrap
import json
import numpy as np
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
import copy

project_path = './'

def generate_embedding(file_name='MCDA5511-classmates - 2025.csv', model='all-MiniLM-L6-v2'):

    attendees_map = {}
    with open(project_path + file_name, newline='') as csvfile:
        attendees = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(attendees)  # Skip the header row
        for row in attendees:
            name, paragraph = row
            attendees_map[paragraph] = name

    # Generate sentence embeddings
    model = SentenceTransformer('sentence-transformers/{}'.format(model))
    paragraphs = list(attendees_map.keys())
    embeddings = model.encode(paragraphs)

    # Create a dictionary to store embeddings for each person
    person_embeddings = {attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}
    return person_embeddings

def compute_similarity_single_to_many(embedding, person_name='SicongFu'):

    query_embedding = embedding[person_name].reshape(1, -1)
    candidate_embeddings = np.array([v for k, v in embedding.items() if k != person_name])
    
    similarities = cosine_similarity(query_embedding, candidate_embeddings)
    return similarities

def compute_spearman_correlation(embeddings1, embeddings2, name='Max Zhao'):

    similarities1 = compute_similarity_single_to_many(embeddings1, name)
    similarities2 = compute_similarity_single_to_many(embeddings2, name)
    print(similarities1)
    ranking1 = np.argsort(-1 * similarities1[0]) + 1
    ranking2 = np.argsort(-1 * similarities2[0]) + 1

    rho, _ = spearmanr(ranking1, ranking2)
    print(ranking1)
    print(ranking2)
    print(f"Spearman’s 𝜌: {rho}")
    return rho

def compute_spearman_average_correlation(embeddings1, embeddings2):

    res = 0
    for k in embeddings1.keys():
        res += compute_spearman_correlation(embeddings1, embeddings2, name=k)
    return res / len(embeddings1.keys())

def do_UMAP(embeddings, parameters, vis_name=None, seed=42):

    reducer = umap.UMAP(n_neighbors=parameters['n_neighbors'], \
                        min_dist=parameters['min_dist'], \
                        random_state=seed,) # \
                        # metric='cosine')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(list(embeddings.values()))
    reduced_data = reducer.fit_transform(scaled_data)
    reduced_embedding = {key: reduced_data[i] for i, (key, value) in enumerate(person_embeddings.items())}

    # print('############', len(reduced_data))

    if vis_name is not None:
        x = [row[0] for row in reduced_data]
        y = [row[1] for row in reduced_data]
        label = list(person_embeddings.keys())

        # Plotting and annotating data points
        plt.figure(figsize=(6, 6)) 
        plt.scatter(x,y)
        for i, name in enumerate(label):
            plt.annotate(name, (x[i], y[i]), fontsize="8")
        plt.axis('off')
        plt.savefig('./{}'.format(vis_name), dpi=800)
        # visualization.png

    return reduced_embedding


if __name__ == '__main__':

    person_embeddings = generate_embedding(model='all-MiniLM-L6-v2')

    # search space
    n_neighbors_list = [5] # [5, 8, 10, 13, 15, 20]
    min_dist_list = [0.9] # [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # best parameter search
    best_score = -np.inf
    best_params = {}
    for n_neighbors, min_dist in product(n_neighbors_list, min_dist_list):
        
        parameters = {'n_neighbors': n_neighbors, 'min_dist': min_dist}
        reduced_data = do_UMAP(person_embeddings, parameters)
        score = compute_spearman_average_correlation(person_embeddings, reduced_data)
        
        if score > best_score:
            best_score = score
            best_params = {'n_neighbors': n_neighbors, 'min_dist': min_dist}
        
        print(f'n_neighbors={n_neighbors}, min_dist={min_dist}, correlation={score:.4f}')

    print(f'Best params: {best_params}, Best score: {best_score:.4f}')

    reduced_data = do_UMAP(person_embeddings, best_params, seed=42, vis_name='vis1.png')
    reduced_data2 = do_UMAP(person_embeddings, best_params, seed=43, vis_name='vis2.png')
    score = compute_spearman_average_correlation(reduced_data2, reduced_data)
    print(f'correlation={score:.4f}')




