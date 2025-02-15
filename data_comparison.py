#!/usr/bin/env python
# coding: utf-8

# #Package Setup

# ## Installations

# In[141]:


import subprocess
import sys

# Installing required packages using subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'umap-learn'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sentence-transformers'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyvis'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'branca'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'seaborn'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])



# ## Imports

# In[142]:


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
import pickle
import numpy as np
import os

project_path = './'


# In[143]:


from sentence_transformers import SentenceTransformer
import csv
project_path='./'


# # Visualization Creation Process

# ## Create Embeddings from Raw Data

# In[144]:


# Read attendees and their responses from a CSV file, replace attendees.csv with own link or file name
attendees_map = {}
project_path = os.path.abspath('./')  # Get the absolute path of the current working directory
file_path = os.path.join(project_path, 'MCDA5511-classmates - 2025.csv')
with open(file_path, newline='') as csvfile:
    attendees = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(attendees)  # Skip the header row
    for row in attendees:
        name, paragraph = row
        attendees_map[paragraph] = name

# Generate sentence embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
paragraphs = list(attendees_map.keys())
embeddings = model.encode(paragraphs)

# Create a dictionary to store embeddings for each person
person_embeddings = {attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}


# In[145]:


with open('person_embeddings.pkl', 'rb') as f:
    load_embeddings = pickle.load(f)
person_embedding1 = load_embeddings["Sriram Ramesh"]
person_embedding2 = person_embeddings["Sriram Ramesh"]

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

similarity = cosine_similarity(person_embedding1, person_embedding2)
print(f"Cosine Similarity for Sriram Ramesh: {similarity}")

person_embedding1 = load_embeddings["Sukanta Dey Amit"]
person_embedding2 = person_embeddings["Sukanta Dey Amit"]

similarity = cosine_similarity(person_embedding1, person_embedding2)
print(f"Cosine Similarity for Sukanta Dey Amit: {similarity}")

person_embedding1 = load_embeddings["Max Zhao"]
person_embedding2 = person_embeddings["Max Zhao"]

similarity = cosine_similarity(person_embedding1, person_embedding2)
print(f"Cosine Similarity for Max Zhao: {similarity}")
    


# In[146]:


# Save embeddings to a pickle file
with open('person_embeddings.pkl', 'wb') as f:
    pickle.dump(person_embeddings, f)


# In[147]:



# In[148]:


with open('person_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
for person, embedding in data.items():
    if isinstance(embedding, np.ndarray):  # Check if the embedding is a numpy array
        data[person] = embedding.tolist()  # Convert to a list
# Save to a JSON file
with open('person_embeddings.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
