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
import warnings

project_path = './'

def plot_umap_embeddings(person_embeddings, model_name='model', save_fig=True):
    names = list(person_embeddings.keys())
    vectors = np.array(list(person_embeddings.values()))
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings_2d = reducer.fit_transform(vectors)
    plt.figure(figsize=(8, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.6)
    for i, name in enumerate(names):
        x, y = embeddings_2d[i, 0], embeddings_2d[i, 1]
        plt.text(x + 0.01, y + 0.01, name, fontsize=9)

    plt.title(f'UMAP Visualization - {model_name}')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    if save_fig:
        plt.savefig(f'umap_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_embedding(file_name='MCDA5511-classmates - 2025.csv', model='all-MiniLM-L6-v2'):
    """
    This function reads data from a CSV file and uses SentenceTransformer to compute the embeddings of the text.
    Each participant has a corresponding text (paragraph), and the embedding is a vector representation of that text.
    Return value
    • person_embeddings (dict):
    • The keys are the participant names and the values are the embedding vectors of their texts (numpy.ndarray).
    """
    
    """
    Read the contents of file_name, each line includes: name (name) and paragraph (text).
    Store it in the attendees_map dictionary, the key is paragraph (text), and the value is name (name).
    """
    attendees_map = {}
    with open(project_path + file_name, newline='') as csvfile:
        attendees = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(attendees)  # Skip the header row
        for row in attendees:
            name, paragraph = row
            attendees_map[paragraph] = name
    
    """
    Use SentenceTransformer to load the model model.
    model.encode(paragraphs) computes the vector representation of all paragraphs (embeddings is a numpy array).
    """
    # Generate sentence embeddings
    model = SentenceTransformer('sentence-transformers/{}'.format(model))
    paragraphs = list(attendees_map.keys())
    embeddings = model.encode(paragraphs)
    
    # The key is the name (attendees_map[paragraph]), and the value is the calculated embedding.
    # Create a dictionary to store embeddings for each person
    person_embeddings = {attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}
    return person_embeddings

def compute_similarity_single_to_many(embedding, person_name='Max Zhao'):
    """
    Calculates the cosine similarity between the specified person and all other people.
    Return value
    • similarities (numpy.ndarray):
    • Shape (1, n-1), representing the cosine similarity between person_name and other n-1 people.
    """
    
    
    """
    1. Get the vector of the target person:
    • embedding[person_name] takes out the vector of person_name and reshapes it to reshape(1, -1) to ensure it is a two-dimensional array.
    2. Get the vectors of all other people:
    • candidate_names takes all people except person_name in embedding.keys().
    • candidate_embeddings takes out the embeddings of these people and converts them into numpy arrays.
    """
    query_embedding = embedding[person_name].reshape(1, -1)
    candidate_names = [k for k in embedding.keys() if k != person_name]
    candidate_embeddings = np.array([embedding[k] for k in candidate_names])
    
    """
    Calculate cosine similarity:
    • cosine_similarity() from sklearn.metrics.pairwise, returns the similarity matrix between query_embedding and candidate_embeddings.
    """
    similarities = cosine_similarity(query_embedding, candidate_embeddings)
    return similarities

def compute_spearman_correlation(embeddings1, embeddings2, name='SicongFu', topN=-1):
    """

    Calculate the Spearman correlation between the embeddings generated by two different models, that is, whether the similarity rankings calculated by embeddings1 and embeddings2 for name are consistent.

    Return value
    • rho (float):
    • rho is the Spearman correlation coefficient, which measures the similarity of two rankings:
        • rho ≈ 1 → highly consistent rankings
        • rho ≈ 0 → irrelevant rankings
        • rho ≈ -1 → completely opposite rankings
    """

    topN = len(embeddings1.keys()) if topN == -1 else topN

    # Calculate the similarity of name under embeddings1 and embeddings2
    similarities1 = compute_similarity_single_to_many(embeddings1, name)
    similarities2 = compute_similarity_single_to_many(embeddings2, name)
    # Ranking by similarity
    ranking1 = np.argsort(-1 * similarities1[0])[: topN] + 1
    ranking2 = np.argsort(-1 * similarities2[0])[: topN] + 1
    print(ranking1)
    print(ranking2)
    # spearmanr(ranking1, ranking2) calculates the correlation of rankings.
    rho, _ = spearmanr(similarities1[0], similarities2[0])
    return rho

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    person_embeddings1 = generate_embedding(model='all-MiniLM-L6-v2')
    person_embeddings2 = generate_embedding(model='all-MiniLM-L12-v2')
    person_embeddings3 = generate_embedding(model='all-mpnet-base-v2')

    names = ['SicongFu'] #, 'Yaxuan Zhang', 'Max Zhao']
    for name in names:

        # rho = compute_spearman_correlation(person_embeddings1, person_embeddings2, name)
        rho = compute_spearman_correlation(person_embeddings1, person_embeddings3, name)
        # rho = compute_spearman_correlation(person_embeddings2, person_embeddings3, name)
        print(f'correlation={rho:.4f}')
    
    plot_umap_embeddings(person_embeddings1, model_name='all-MiniLM-L6-v2')
    plot_umap_embeddings(person_embeddings2, model_name='all-MiniLM-L12-v2')
    plot_umap_embeddings(person_embeddings3, model_name='all-mpnet-base-v2')




