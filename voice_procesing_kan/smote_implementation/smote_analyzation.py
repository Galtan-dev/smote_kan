import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import torch
from statistics import mean
from kan import KAN
import os

# parametrs
kbs = 100
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_set_name = "smote_100_speed_100"
# txt_features_output = "E:\\PyCharmProjects\\voice_procesing_kan\\smote_implementation\\features_output.txt"
# txt_path_output = "E:\\PyCharmProjects\\voice_procesing_kan\\smote_implementation\\status_output.txt"

txt_features_output = "/disk2/seinerj/smote_kan/features_output.txt"
txt_path_output = "/disk2/seinerj/smote_kan/status_output.txt"


# data loading
with open(txt_path_output, 'r') as file:
    statuses = file.read()
    statuses = statuses.strip().replace("\n", ",")
    statuses = statuses.split(",")

with open(txt_features_output, "r") as file:
    rows = [list(map(float, line.strip().split(","))) for line in file]


# data standardization
data_matrix = np.array(rows)
scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = MinMaxScaler()
data_standardized = scaler.fit_transform(data_matrix)
standardized_rows = data_standardized.tolist()


# train and test data separation
data_with_status = list(zip(standardized_rows, statuses))
healthy_and_unhealthy = [item for item in data_with_status if item[1] in ("healthy", "unhealthy")]
augmentations = [item for item in data_with_status if item[1] in ("noise", "shifted", "speed", "synthetic")]
total_samples = len(healthy_and_unhealthy)
percent_20 = int(0.2 * total_samples)
random.shuffle(healthy_and_unhealthy)
test_samples = healthy_and_unhealthy[:percent_20]
train_samples = healthy_and_unhealthy[percent_20:]
train_samples.extend(augmentations)
test_data = [item[0] for item in test_samples]
train_data = [item[0] for item in train_samples]
test_labels = [item[1] for item in test_samples]
train_labels = [item[1] for item in train_samples]
test_labels = [1 if status == "healthy" or status not in ("healthy", "unhealthy")
               else 0 for status in test_labels]
train_labels = [1 if status == "healthy" or status not in ("healthy", "unhealthy")
                else 0 for status in train_labels]


# features selection - original approach
selector = SelectKBest(score_func=mutual_info_classif, k=kbs)
train_features = selector.fit_transform(train_data, train_labels)
test_features = selector.transform(test_data)

# # alternative aproach
# selector = SelectKBest(score_func=mutual_info_classif, k=kbs)
# train_features = selector.fit_transform(train_data, train_labels)
# test_features = selector.fit_transform(test_data, test_labels)


# dataset preparation
train_tensors = torch.tensor(train_features)
train_labels = torch.tensor(train_labels)
test_tensors = torch.tensor(test_features)
test_labels = torch.tensor(test_labels)

# Vytvoření slovníku s daty
input_dataset = {
    'train_input': train_tensors.float().to(device),
    'train_label': train_labels.long().to(device),
    'test_input': test_tensors.float().to(device),
    'test_label': test_labels.long().to(device)
}


def auto_res_log(results, kan_arch, grid, k, ksb, steps, lamb):
    # Výpočet průměru z posledních 15 F1 skóre
    if len(results["test_uar"]) >= 15:
        avg_last_15_f1 = mean(results["test_uar"][-15:])
    else:
        avg_last_15_f1 = mean(results["test_uar"])  # Použijeme všechny hodnoty, pokud je jich méně než 15

    # Poslední hodnota F1 skóre
    last_f1 = results["test_uar"][-1]

    # Průměr a maximální hodnota UAR
    avg_uar = mean(results["test_uar"])
    max_val = max(results["test_uar"])

    # Cesta k adresáři a souboru
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)  # Vytvoření adresáře, pokud neexistuje
    file_path = os.path.join(log_dir, f"{data_set_name}.txt")

# Zapisování všech epoch do souboru
def auto_res_log(results, kan_arch, grid, k, ksb, steps, lamb, train_labels, test_labels):
    # Výpočet celkového počtu pozitivních a negativních vzorků
    train_positives = sum(train_labels)
    train_negatives = len(train_labels) - train_positives
    test_positives = sum(test_labels)
    test_negatives = len(test_labels) - test_positives

    # Cesta k adresáři a kontrola, jaké soubory již existují
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)  # Vytvoření adresáře, pokud neexistuje

    # Vytvoření pořadového čísla souboru
    existing_files = [f for f in os.listdir(log_dir) if f.startswith(data_set_name)]
    file_number = len(existing_files) + 1
    file_path = os.path.join(log_dir, f"{data_set_name}_{file_number}.txt")

    # Zapisování všech epoch do souboru
    with open(file_path, "w") as file:
        file.write(
            f"Arch: {kan_arch}, Grid: {grid}, k: {k}, ksb: {ksb}, Steps: {steps}, Lambda: {lamb}\n"
        )
        file.write(f"{'Epoch':<8}{'UAR':<10}{'TP':<10}{'TN':<10}{'FP':<10}{'FN':<10}\n")
        for epoch, (uar, tp, tn, fp, fn) in enumerate(zip(
                results["test_uar"],
                results["test_tp"],
                results["test_tn"],
                results["test_fp"],
                results["test_fn"]
        ), start=1):
            file.write(f"{epoch:<8}{uar:<10.4f}{tp:<10}{tn:<10}{fp:<10}{fn:<10}\n")

        # Zapsání počtu pozitivních a negativních vzorků
        file.write("\nDataset Statistics:\n")
        file.write(f"Training Positives: {train_positives}, Training Negatives: {train_negatives}\n")
        file.write(f"Testing Positives: {test_positives}, Testing Negatives: {test_negatives}\n")

def test_tp():
    predictions = torch.argmax(model(input_dataset["test_input"].to(device)), dim=1)
    labels = input_dataset["test_label"].to(device)
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    return tp

def test_tn():
    predictions = torch.argmax(model(input_dataset["test_input"].to(device)), dim=1)
    labels = input_dataset["test_label"].to(device)
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    return tn

def test_fp():
    predictions = torch.argmax(model(input_dataset["test_input"].to(device)), dim=1)
    labels = input_dataset["test_label"].to(device)
    fp = ((predictions == 1) & (labels == 0)).sum().float()
    return fp

def test_fn():
    predictions = torch.argmax(model(input_dataset["test_input"].to(device)), dim=1)
    labels = input_dataset["test_label"].to(device)
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    return fn

def test_uar():
    predictions = torch.argmax(model(input_dataset["test_input"].to(device)), dim=1)
    labels = input_dataset["test_label"].to(device)
    tn = ((predictions == 0) & (labels == 0)).sum().float()
    tp = ((predictions == 1) & (labels == 1)).sum().float()
    fn = ((predictions == 0) & (labels == 1)).sum().float()
    fp = ((predictions == 1) & (labels == 0)).sum().float()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    uar = 0.5 * (recall + specificity)
    return uar

# Funkce pro generování architektur s možností přeskočení
def kan_arch_gen(input_size, skip=0):
    steps = list(np.linspace(0, 2, 11))
    kan_archs = []

    # Procházení všech kombinací vrstev
    for first in steps:
        first_layer = input_size * 2 - int(first * input_size)
        if first_layer > 0:
            kan_archs.append([[input_size, 0], [first_layer, 0], [2, 3]])

        for second in steps:
            second_layer = input_size * 2 - int(second * input_size)
            if first_layer >= second_layer > 0:
                kan_archs.append([[input_size, 0], [first_layer, 0], [second_layer, 0], [2, 3]])

    # Vrácení pouze architektur po přeskočení daného počtu
    return kan_archs[skip:]

skip_archs = 0
steps = 100
lamb = 0.001
ratio = 0.8
grids = [5, 6]
k_set = [4, 3]
kan_arch = kan_arch_gen(kbs, skip=skip_archs)


# # [[100, 0], [100, 0], [100, 0], [2, 3]],6,4,100,120,0.001
# kan_arch = [[[100, 0], [160, 0], [100, 0], [2, 3]]]
# grids = [5]
# k_set = [3]
# steps = 50


for k in k_set:
    for arch in kan_arch:
        for grid in grids:
            print(f"Experiment: {arch},{grid},{k},{kbs},{steps},{lamb}\n")
            model = KAN(width=arch, grid=grid, k=k, device=device, seed=seed, auto_save=False, save_act=True)
            results = model.fit(
                input_dataset,
                opt="LBFGS",
                steps=steps,
                lamb=lamb,
                metrics=(test_fn, test_fp, test_tn, test_tp, test_uar),
                loss_fn=torch.nn.CrossEntropyLoss()
                   )
            auto_res_log(results, arch, grid, k, kbs, steps, lamb, train_labels.tolist(), test_labels.tolist())
