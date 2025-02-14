import csv
import umap
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from textwrap import wrap
import numpy as np
from scipy.stats import spearmanr
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import optuna
from sklearn.neighbors import NearestNeighbors

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

    ranking1 = np.argsort(-1 * similarities1[0]) + 1
    ranking2 = np.argsort(-1 * similarities2[0]) + 1

    rho, _ = spearmanr(similarities1[0], similarities2[0])

    # print(ranking1)
    # print(ranking2)
    # print(f"Spearman’s 𝜌: {rho}")
    return rho

def compute_spearman_average_correlation(embeddings1, embeddings2):

    res = 0
    for k in embeddings1.keys():
        res += compute_spearman_correlation(embeddings1, embeddings2, name=k)
    return res / len(embeddings1.keys())

def do_UMAP(embeddings, parameters=None, vis_name=None, seed=42, do_kmeans=False, mark_wrong=False):

    if parameters is not None:
        reducer = umap.UMAP(n_neighbors=parameters['n_neighbors'], \
                            min_dist=parameters['min_dist'], \
                            random_state=seed,
                            spread=parameters['spread'],
                            metric='euclidean',
                            n_components=2)
    else:
        reducer = umap.UMAP(random_state=seed,
                            metric='euclidean',
                            n_components=2)     
           
    with np.errstate(divide='ignore'):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(list(embeddings.values()))
        reduced_data = reducer.fit_transform(scaled_data)
        # print(reduced_data)
        reduced_embedding = {key: reduced_data[i] for i, (key, value) in enumerate(embeddings.items())}

    if vis_name is not None:
        x = [row[0] for row in reduced_data]
        y = [row[1] for row in reduced_data]
        label = list(embeddings.keys())

        # Plotting and annotating data points
        plt.figure(figsize=(12, 12))
        if do_kmeans:
            num_clusters = 3 
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(reduced_data)
            colors = plt.cm.get_cmap("tab10", num_clusters)
            for i in range(num_clusters):
                cluster_points = [j for j in range(len(cluster_labels)) if cluster_labels[j] == i]
                plt.scatter([x[j] for j in cluster_points], 
                            [y[j] for j in cluster_points], 
                            s=100, 
                            color=colors(i), 
                            label=f'Cluster {i}')
        elif mark_wrong:
            topN = 5
            original_neighbors = NearestNeighbors(n_neighbors=topN+1, metric='cosine').fit(scaled_data)
            _, original_indices = original_neighbors.kneighbors(scaled_data)
            reduced_neighbors = NearestNeighbors(n_neighbors=topN+1, metric='euclidean').fit(reduced_data)
            _, reduced_indices = reduced_neighbors.kneighbors(reduced_data)

            incorrect_points = []
            correct_points = []
            for i in range(len(reduced_data)):

                original_set = set(original_indices[i][1:])
                reduced_set = set(reduced_indices[i][1:])

                print(original_set)
                print(reduced_set)
                intersection = len(original_set & reduced_set)
                union = len(original_set | reduced_set)
                print(intersection / union)
                if intersection / union < 0.5:
                    incorrect_points.append(i)
                else:
                    correct_points.append(i)

            # import pdb; pdb.set_trace()

            if incorrect_points:
                plt.scatter(reduced_data[incorrect_points, 0], reduced_data[incorrect_points, 1], s=100, color='red')
            if correct_points:
                plt.scatter(reduced_data[correct_points, 0], reduced_data[correct_points, 1], s=100)
        else:
            plt.scatter(x, y, s=100)

        for i, name in enumerate(label):
            plt.annotate(name, (x[i], y[i]), fontsize="20")
        plt.axis('off')
        try:
            plt.savefig('./{}'.format(vis_name), dpi=800)
        except:
            return reduced_embedding

    return reduced_embedding

def grid_search(embeddings):

    # search space
    n_neighbors_list = [5] #[10] [14] # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    min_dist_list = [0.25] # [0.9] # [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9]
    spread_list = [8] # [9] # [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # n_neighbors_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # min_dist_list = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9]
    # spread_list = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # best parameter search
    best_score = -np.inf
    best_params = {}
    for n_neighbors, min_dist, spread in product(n_neighbors_list, min_dist_list, spread_list):

        if min_dist > spread:
            continue
        
        parameters = {'n_neighbors': n_neighbors, 'min_dist': min_dist, 'spread':spread}
        reduced_data = do_UMAP(embeddings, parameters, seed=42)
        score = compute_spearman_average_correlation(embeddings, reduced_data)
        
        if score > best_score:
            best_score = score
            best_params = {'n_neighbors': n_neighbors, 'min_dist': min_dist, 'spread':spread}
        
        print(f'n_neighbors={n_neighbors}, min_dist={min_dist}, correlation={score:.4f}')

    print(f'Best params: {best_params}, Best score: {best_score:.4f}')
    return best_params

def optuna_search(embeddings):
    def objective(trial):

        n_neighbors = trial.suggest_int("n_neighbors", 2, 20)
        min_dist = trial.suggest_float("min_dist", 0.0, 0.99)
        spread = trial.suggest_float("spread", min_dist, 10)

        reduced_embedding = do_UMAP(person_embeddings, \
                                    parameters={'n_neighbors':n_neighbors, 'min_dist':min_dist, 'spread':spread}, \
                                    seed=42)
        score = compute_spearman_average_correlation(embeddings, reduced_embedding)
        return score
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=300)
    print("Best Parameters:", study.best_params)
    return study.best_params


if __name__ == '__main__':

    person_embeddings = generate_embedding(model='all-MiniLM-L6-v2')

    # best_params = optuna_search(person_embeddings)
    best_params = grid_search(person_embeddings)

    reduced_data1 = do_UMAP(person_embeddings, seed=42, vis_name='vis1_ori.png', mark_wrong=True)
    reduced_data2 = do_UMAP(person_embeddings, seed=52, vis_name='vis2_ori.png', mark_wrong=True)
    reduced_data3 = do_UMAP(person_embeddings, seed=32, vis_name='vis3_ori.png', mark_wrong=True)
    score1 = compute_spearman_average_correlation(person_embeddings, reduced_data1)
    print(f'correlation={score1:.4f}')
    score2 = compute_spearman_average_correlation(person_embeddings, reduced_data2)
    print(f'correlation={score2:.4f}')
    score3 = compute_spearman_average_correlation(person_embeddings, reduced_data3)
    print(f'correlation={score3:.4f}') 
    
    
    reduced_data1 = do_UMAP(person_embeddings, best_params, seed=42, vis_name='vis1_tune.png', mark_wrong=True)
    reduced_data2 = do_UMAP(person_embeddings, best_params, seed=52, vis_name='vis2_tune.png', mark_wrong=True)
    reduced_data3 = do_UMAP(person_embeddings, best_params, seed=32, vis_name='vis3_tune.png', mark_wrong=True)
    score1 = compute_spearman_average_correlation(person_embeddings, reduced_data1)
    print(f'correlation={score1:.4f}')
    score2 = compute_spearman_average_correlation(person_embeddings, reduced_data2)
    print(f'correlation={score2:.4f}')
    score3 = compute_spearman_average_correlation(person_embeddings, reduced_data3)
    print(f'correlation={score3:.4f}')


    # 读取图片
    image_files = [
        'vis1_ori.png', 'vis2_ori.png', 'vis3_ori.png',  # 第一行：默认参数
        'vis1_tune.png', 'vis2_tune.png', 'vis3_tune.png'  # 第二行：最佳参数
    ]
    images = [mpimg.imread(img_file) for img_file in image_files]

    # 确保所有图片的格式一致
    images = [img if img.dtype == np.uint8 else (img * 255).astype(np.uint8) for img in images]

    # 获取图像尺寸
    img_height, img_width, img_channels = images[0].shape

    # 创建空白画布 (2行 x 3列)
    combined_image = np.zeros((2 * img_height, 3 * img_width, img_channels), dtype=np.uint8)

    # 填充画布
    for i in range(2):
        for j in range(3):
            combined_image[i * img_height: (i + 1) * img_height, 
                        j * img_width: (j + 1) * img_width, :] = images[i * 3 + j]

    # 确保是 uint8 格式
    from PIL import Image
    combined_image = combined_image.astype(np.uint8)

    # **使用 PIL 进行缩放至 1080x720**
    target_size = (1080, 720)  # (width, height)
    resized_image = Image.fromarray(combined_image).resize(target_size, Image.LANCZOS)

    # 保存缩小后的图像
    output_filename = "combined_umap_resized.png"
    resized_image.save(output_filename)
    




