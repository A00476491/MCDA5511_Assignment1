import csv
import umap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn_extra.cluster import KMedoids
from sentence_transformers import SentenceTransformer
import warnings
import matplotlib.image as mpimg
from tabulate import tabulate

# Suppress warnings
warnings.simplefilter("ignore")
np.random.seed(42)

# Define project path
PROJECT_PATH = './'


def generate_embedding(file_name='MCDA5511-classmates - 2025.csv', model_name='all-MiniLM-L6-v2'):
    """
    Reads a CSV file containing names and interst descriptions, generates sentence embeddings using a transformer model,
    and returns a dictionary mapping names to their corresponding embeddings.
    """

    # read csv file
    attendees_map = {}
    with open(PROJECT_PATH + file_name, newline='') as csvfile:
        attendees = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(attendees)
        for row in attendees:
            name, paragraph = row
            attendees_map[paragraph] = name

    # Generate sentence embeddings
    model = SentenceTransformer('sentence-transformers/{}'.format(model_name))
    paragraphs = list(attendees_map.keys())
    embeddings = model.encode(paragraphs)

    # Create a dictionary to store embeddings for each person
    person_embeddings = {attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}
    return person_embeddings

def compute_similarity_single_to_many(embedding, person_name='SicongFu'):
    """
    Computes cosine similarity between a single person's embedding and all other embeddings.
    """
    query_embedding = embedding[person_name].reshape(1, -1)
    candidate_embeddings = np.array([v for k, v in embedding.items() if k != person_name])
    
    return cosine_similarity(query_embedding, candidate_embeddings)

def compute_spearman_correlation(embeddings1, embeddings2, name='Max Zhao'):
    """
    Computes the Spearman rank correlation between two embedding sets for a given person.
    """
    similarities1 = compute_similarity_single_to_many(embeddings1, name)
    similarities2 = compute_similarity_single_to_many(embeddings2, name)
    
    rho, _ = spearmanr(similarities1[0], similarities2[0])
    return rho

def compute_spearman_average_correlation(embeddings1, embeddings2):
    """
    Computes the average Spearman rank correlation for all names.
    """
    return np.mean([compute_spearman_correlation(embeddings1, embeddings2, name=k) for k in embeddings1.keys()])

def do_UMAP(embeddings, parameters=None, vis_name=None, seed=42, cluster_labels=None):
    """
    Performs dimensionality reduction using UMAP and optionally visualizes the result.
    """

    if parameters is None:
        reducer = umap.UMAP(
            random_state=seed,
            metric='euclidean',
            n_components=2
        )
    else:
        reducer = umap.UMAP(
            n_neighbors=parameters['n_neighbors'],
            min_dist=parameters['min_dist'],
            spread=parameters['spread'],
            random_state=seed,
            metric='euclidean',
            n_components=2
        )
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(list(embeddings.values()))
    reduced_data = reducer.fit_transform(scaled_data)
    reduced_embedding = {key: reduced_data[i] for i, key in enumerate(embeddings.keys())}
    
    if vis_name:
        x, y = reduced_data[:, 0], reduced_data[:, 1]
        labels = list(embeddings.keys())
        plt.figure(figsize=(12, 12))
        
        if cluster_labels is not None:
            colors = matplotlib.colormaps["tab10"]
            for i in range(3):
                cluster_points = [j for j in range(len(cluster_labels)) if cluster_labels[j] == i]
                plt.scatter(x[cluster_points], y[cluster_points], s=300, color=colors(i), label=f'Cluster {i}')
        else:
            plt.scatter(x, y, s=300)
        
        for i, name in enumerate(labels):
            plt.annotate(name.split()[0], (x[i], y[i]), fontsize=24)
        
        plt.axis('off')
        plt.savefig(vis_name, dpi=800)
    
    return reduced_embedding

def grid_search(embeddings, search_space):
    """
    Performs a grid search to find the best UMAP parameters based on Spearman correlation.
    """

    best_score = -np.inf
    best_params = {}
    
    # Iterate through all combinations of parameters
    for n_neighbors, min_dist, spread in product(
            search_space['n_neighbors'], search_space['min_dist'], search_space['spread']):
        
        if min_dist > spread:
            continue  # Ensure valid parameter settings
        
        parameters = {'n_neighbors': n_neighbors, 'min_dist': min_dist, 'spread': spread}
        reduced_data = do_UMAP(embeddings, parameters, seed=42)
        score = compute_spearman_average_correlation(embeddings, reduced_data)
        
        if score > best_score:
            best_score = score
            best_params = {'n_neighbors': n_neighbors, 'min_dist': min_dist, 'spread':spread}
    
    print(f'Best params: {best_params}, Best score: {best_score:.4f}')
    return best_params

def tabulate_output(results):

    headers = ["Seed 42", "Seed 52", "Seed 72"]
    rows = [
        ["Default Parameters"] + results[0],
        ["Best Parameters"] + results[1],
    ]

    table = tabulate(rows, headers=headers, tablefmt="grid")
    print(table)

def matrix_charts():

    image_files = [
        "./vis/umap_default_parameters_seed_42.png",
        "./vis/umap_default_parameters_seed_52.png",
        "./vis/umap_default_parameters_seed_72.png",
        "./vis/umap_best_parameters_seed_42.png",
        "./vis/umap_best_parameters_seed_52.png",
        "./vis/umap_best_parameters_seed_72.png",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    seed_values = ["Seed 42", "Seed 52", "Seed 72"]
    param_types = ["Default Parameters", "Best Parameters"]

    for i, seed in enumerate(seed_values):
        axes[0, i].set_title(seed, fontsize=14, pad=20)
    fig.text(0.06, 0.75, "Default Parameters", va='center', ha='center', fontsize=14, rotation=90)
    fig.text(0.06, 0.25, "Best Parameters", va='center', ha='center', fontsize=14, rotation=90)
    
    
    for i, param in enumerate(param_types):
        for j, seed in enumerate(seed_values):
            idx = i * 3 + j 
            img = mpimg.imread(image_files[idx])
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis("off")

            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2) 

    plt.tight_layout()
    plt.savefig("./vis/umap_comparison.png", dpi=800, bbox_inches="tight")

if __name__ == '__main__':
    # Step1: Generate embeddings
    person_embeddings = generate_embedding(model_name='all-MiniLM-L6-v2')
    
    # Step2: Apply clustering by Cosine similarity
    kmedoids = KMedoids(n_clusters=3, metric="cosine", random_state=42)
    cluster_labels = kmedoids.fit_predict(list(person_embeddings.values()))
    
    # Step3: Perform UMAP with default parameters for different seeds
    umap_seeds = [42, 52, 72]
    results = [[], []]
    for umap_seed in umap_seeds:
        reduced_data = do_UMAP(
            person_embeddings, 
            vis_name=f'./vis/umap_default_parameters_seed_{umap_seed}.png', 
            seed=umap_seed, 
            cluster_labels=cluster_labels
        )
        avg_correlation = compute_spearman_average_correlation(person_embeddings, reduced_data)
        # print(f'correlation={avg_correlation:.4f} for seed={umap_seed} with default parameters')
        results[0].append(avg_correlation)
    
    # Step4: Perform grid search to optimize UMAP parameters
    search_space = {
        'n_neighbors': [10], # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'min_dist': [0.25], # [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9],
        'spread': [8], # [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    best_params = grid_search(person_embeddings, search_space)
     
    # Step5: Perform UMAP with best parameters for different seeds
    umap_seeds = [42, 52, 72]
    for umap_seed in umap_seeds:
        reduced_data = do_UMAP(
            person_embeddings, 
            best_params, 
            vis_name=f'./vis/umap_best_parameters_seed_{umap_seed}.png', 
            seed=umap_seed, 
            cluster_labels=cluster_labels
        )
        avg_correlation = compute_spearman_average_correlation(person_embeddings, reduced_data)
        # print(f'correlation={avg_correlation:.4f} for seed={umap_seed} with best parameters')
        results[1].append(avg_correlation)
    tabulate_output(results)
    matrix_charts()
    




