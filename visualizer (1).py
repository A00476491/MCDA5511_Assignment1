#!/usr/bin/env python
# coding: utf-8

# #Package Setup

# ## Installations

# In[141]:


get_ipython().system('pip install umap-learn')
get_ipython().system('pip install sentence-transformers')
get_ipython().system('pip install pyvis')
get_ipython().system('pip install branca')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install --upgrade pip')


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
with open(project_path+ 'MCDA5511-classmates - 2025.csv', newline='') as csvfile:
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


#print (person_embeddings["Pawan Lingras"])


# In[148]:


with open('person_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
for person, embedding in data.items():
    if isinstance(embedding, np.ndarray):  # Check if the embedding is a numpy array
        data[person] = embedding.tolist()  # Convert to a list
# Save to a JSON file
with open('person_embeddings.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)


# ## Reducing dimensionality of embedding data, scaling to coordinate domain/range
# 

# In[149]:


# Reducing dimensionality of embedding data, scaling to coordinate domain/range
reducer = umap.UMAP(random_state=42) # set the seed to 42
scaler = StandardScaler()
scaled_data = scaler.fit_transform(list(person_embeddings.values()))
reduced_data = reducer.fit_transform(scaled_data)


# ## Create Visualization Image

# In[150]:


# Creating lists of coordinates with accompanying labels
x = [row[0] for row in reduced_data]
y = [row[1] for row in reduced_data]
label = list(person_embeddings.keys())

# Plotting and annotating data points
plt.scatter(x,y)
for i, name in enumerate(label):
    plt.annotate(name, (x[i], y[i]), fontsize="8")

# Clean-up and Export
plt.axis('off')
plt.savefig(project_path+'visualization.png', dpi=800)


# ## Find the  top N matches to a node

# In[140]:


# Providing top matches
top_matches = {}
all_personal_pairs = defaultdict(list)
for person in attendees_map.values():
    for person1 in attendees_map.values():
        all_personal_pairs[person].append([spatial.distance.cosine(person_embeddings[person1], person_embeddings[person]), person1])

n = 5
# Collect the top n most similarity nodes
data_day_list = []
for person in attendees_map.values():
    top_matches[person] = sorted(all_personal_pairs[person], key=lambda x: x[0])[1:n+1] # drop yourself, take top 5
    a = sorted(all_personal_pairs[person], key=lambda x: x[0])[1:n+1]
    mini_df = pd.DataFrame(a, columns=['Weight', 'Target'])
    mini_df['Source'] = person
    data_day_list.append(mini_df)

# output this information as a json
with open(project_path + 'b_top5_matches.json', 'w') as json_file:
    json.dump(top_matches, json_file)

# Output this information as a csv
df = pd.concat(data_day_list)
df.to_csv(project_path + 'b_top5_matches.csv')


# ## Add Colour/Paragraph Information to Dataframe

# In[ ]:


# Get the colour pallette
colour = sns.color_palette("pastel",len(x)).as_hex()

# Add colour pallette to the df
df1 = pd.DataFrame([label,colour])
df1 = df1.T
df1.rename(columns={0: 'Source', 1: 'Colour'},inplace=True)
df = df.set_index('Source').join(df1.set_index('Source'))
df['Source'] = df.index
df = df.reset_index(drop=True)

# Add colour pallette for both the df Target and Source:
df1.rename(columns={'Source': 'Target'},inplace=True)
df = df.set_index('Target').join(df1.set_index('Target'),lsuffix='_Source', rsuffix='_Target')
df['Target'] = df.index
df = df.reset_index(drop=True)
print(df)

# Add paragraphs to the df
df2 = pd.DataFrame([label,paragraphs])
df2 = df2.T
df2.rename(columns={0: 'Source', 1: 'Paragraphs'},inplace=True)
df = df.set_index('Source').join(df2.set_index('Source'))
df['Source'] = df.index
df = df.reset_index(drop=True)
print(df)

# Create a cleaned Dataframe of just the Source and and Paragraph information
df_new = df[["Source","Paragraphs"]]
df_new = df_new.drop_duplicates()
df_new.set_index('Source', inplace=True)


# ## Build Interative Network Visualization (Simple)

# In[ ]:


# Intitalize bucket size and colour palettes
buckets = [100] * len(x)
colour = sns.color_palette("pastel",len(x)).as_hex()

# Initialize network
g = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

# Add unconnected nodes to the network
g.add_nodes(list(range(1,len(x)+1)), value=buckets,
                         title=paragraphs,
                         x=np.array(x).astype(np.float64),
                         y=np.array(y).astype(np.float64),
                         label=label,
                         color=colour)

# Output the visualization
g.toggle_physics(True)
g.show(project_path+'c_simple_viz.html', notebook=False)


# ## Build Interative Network Visualization (Complex)
# 

# In[ ]:


# Initialize network
got_net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white",select_menu=True,cdn_resources='remote')

# Create a dictionary of Important information
sources = df['Source']
targets = df['Target']
weights = df['Weight']
color_targets = df['Colour_Target']
color_sources = df['Colour_Source']

edge_data = zip(sources, targets, weights,color_targets,color_sources)

# Add nodes and edges to the network
for e in edge_data:
                src = e[0]
                dst = e[1]
                w = e[2]
                c_t= e[3]
                c_s= e[4]
                got_net.add_node(src, src, title=src,color=c_s)
                got_net.add_node(dst, dst, title=dst)
                got_net.add_edge(src, dst, value=w)#,color = "#c79910") # if you  want a solide colour for edges

# Add paragraphs to the popup
for i,node in enumerate(got_net.nodes):
               content =df_new.loc[node.get("title"),"Paragraphs"]
               node["title"] += ": "+ "\n \n" +'\n'.join(wrap(content, width=50))

## Output the visualization
got_net.show(project_path+'c_complex_viz.html', notebook=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




