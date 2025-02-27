import sys, csv, json
import numpy as np

def calculate_similarity(sim_check, x, y):
    match(sim_check):
        case "cosine similarity":
            return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
        case "hamming distance":
            return N - sum(int(xi) ^ int(yi) for xi, yi in zip(x, y))
        case "dot product":
            return np.dot(x,y)
        
def generate_hdv(element_type, n):

    match(element_type):
        case "bipolar":
            return np.random.choice([-1, 1], size=n)
        case "binary":
            return np.random.choice([0, 1], size=n)
        case "float":
            return np.random.uniform(-1, 1, size=n)
        

# bind is unnecessary in my implementation        
def bind(element_type, hdv_list: list):
    hdv_bind = np.empty(len(hdv_list[0]))
    match(element_type):
        case "bipolar":
            pass
        case "binary":
            pass
        case "float":
            pass

def bundle(element_type, hdv_list: list):
    if (len(hdv_list) % 2) == 0:
        hdv_temp = generate_hdv(element_type, len(hdv_list[0]))
        hdv_list.append(hdv_temp)

    hdv_bundle = np.empty(len(hdv_list[0]))
    match(element_type):
        case "bipolar":
            sum = 0
            for i in range(len(hdv_list[0])):
                for hdv in hdv_list:
                    sum += hdv[i]
                hdv_bundle[i] = 1 if sum > 0 else -1
                sum = 0
            return hdv_bundle
        case "binary":
            sum = 0
            for i in range(len(hdv_list[0])):
                for hdv in hdv_list:
                    sum += hdv[i]
                hdv_bundle[i] = sum > len(hdv_list)/2
                sum = 0
            return hdv_bundle
        case "float":
            sum = 0
            for i in range(len(hdv_list[0])):
                for hdv in hdv_list:
                    sum += hdv[i]
                if sum > 1.0:
                    sum = 1.0
                elif sum < -1.0:
                    sum = -1.0
                hdv_bundle[i] = sum
                sum = 0
            return hdv_bundle

if __name__ == "__main__":

    config_file_path = sys.argv[1] if len(sys.argv) > 1 else '../config/hdc_config.json'
    csv_file_path = '../data/hdc_roles_fillers.csv'
    with open(config_file_path, 'r') as config_file:
        config_data = json.load(config_file)

    N = config_data["hypervector_dimension"]
    element_type = config_data["element_type"]
    sim_check = config_data["element_type_options"][element_type]["similarity_check"]

    num_features = config_data["number_of_features"]        
    query_source = config_data["query_source"]
    query_source_hdv = generate_hdv(element_type, N)
    query_target = config_data["query_target"]
    query_target_hdv = generate_hdv(element_type, N)

    num_iterations = config_data["number_of_runs"]

    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)[:num_features]
        feature_data = [row[:num_features] for row in reader] 

    feature_index_x = -1
    feature_index_y = -1

    for row in feature_data:
        for feature in row:
            if feature == query_source:
                feature_index_x = row.index(feature)
            if feature == query_target:
                feature_index_y = feature_data.index(row)
    
    if ((feature_index_x == -1) or (feature_index_y == -1)):
        print("Invalid query")
        exit(1)

    correct_result = feature_data[feature_index_y][feature_index_x]

    hdv_matrix = []

    for row in feature_data:
        hdv_row = []
        for feature in row:
            if (feature == query_source):
                hdv_row.append(query_source_hdv)
            if (feature == query_target):
                hdv_row.append(query_target_hdv)
            else:
                hdv_row.append(generate_hdv(element_type, N))
        hdv_matrix.append(hdv_row)
        hdv_row = []
    
    hdv_matrix_tranpose = list(zip(*hdv_matrix))

    hdv_bundled_rows = []
    hdv_bundled_cols = []


    for row in hdv_matrix:
        hdv_bundled_rows.append(bundle(element_type, row))
    for col in hdv_matrix_tranpose:
        hdv_bundled_cols.append(bundle(element_type, col))

    similarities = []
    for col in hdv_bundled_cols:
        similarities.append(calculate_similarity(sim_check, query_source_hdv, col))

    x_coord = similarities.index(max(similarities))

    similarities = []

    for row in hdv_bundled_rows:
        similarities.append(calculate_similarity(sim_check, query_target_hdv, row))
    
    y_coord = similarities.index(max(similarities))

    print(f"Query: What is the {query_source} of {query_target} ? {feature_data[y_coord][x_coord]}", flush=True)
    print(f"Evaluation: Evaluating for {num_iterations} runs with {num_features} role/filler pairs bundled in a {element_type} prototype hypervector with {N} elements", flush=True)

    num_successes = 0

    for i in range(num_iterations):
        hdv_matrix = []
        for row in feature_data:
            hdv_row = []
            for feature in row:
                if (feature == query_source):
                    hdv_row.append(query_source_hdv)
                if (feature == query_target):
                    hdv_row.append(query_target_hdv)
                else:
                    hdv_row.append(generate_hdv(element_type, N))
            hdv_matrix.append(hdv_row)
            hdv_row = []
        
        hdv_matrix_tranpose = list(zip(*hdv_matrix))

        hdv_bundled_rows = []
        hdv_bundled_cols = []


        for row in hdv_matrix:
            hdv_bundled_rows.append(bundle(element_type, row))
        for col in hdv_matrix_tranpose:
            hdv_bundled_cols.append(bundle(element_type, col))

        similarities = []
        for col in hdv_bundled_cols:
            similarities.append(calculate_similarity(sim_check, query_source_hdv, col))
        
        x_coord = similarities.index(max(similarities))

        similarities = []

        for row in hdv_bundled_rows:
            similarities.append(calculate_similarity(sim_check, query_target_hdv, row))
        
        y_coord = similarities.index(max(similarities))

        result = feature_data[y_coord][x_coord]

        if (result == correct_result):
            num_successes += 1

        print("=", end="", flush=True)
    
    print(f"\nSuccess rate is {float(num_successes)/float(num_iterations)*100}")






