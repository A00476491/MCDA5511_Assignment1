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

def compute_similarity_single_to_many(embedding, person_name='Max Zhao'):

    query_embedding = embedding[person_name].reshape(1, -1)
    candidate_embeddings = np.array([v for k, v in embedding.items() if k != person_name])
    
    similarities = cosine_similarity(query_embedding, candidate_embeddings)
    return similarities

def compute_spearman_correlation(embeddings1, embeddings2, name='Max Zhao'):

    similarities1 = compute_similarity_single_to_many(embeddings1, name)
    similarities2 = compute_similarity_single_to_many(embeddings2, name)
    ranking1 = np.argsort(similarities1[0]) + 1
    ranking2 = np.argsort(similarities2[0]) + 1

    rho, _ = spearmanr(ranking1, ranking2)
    print(f"Spearman’s 𝜌: {rho}")
    return rho

if __name__ == '__main__':

    person_embeddings1 = generate_embedding(model='all-MiniLM-L6-v2')
    person_embeddings2 = generate_embedding(model='all-MiniLM-L12-v2')

    names = ['Max Zhao', 'Yaxuan Zhang', 'SicongFu']
    for name in names:

        rho = compute_spearman_correlation(person_embeddings1, person_embeddings2, name)
        print(f'correlation={rho:.4f}')




