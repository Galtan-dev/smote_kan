import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import random

seed = 0
np.random.seed(seed)
random.seed(seed)

# Funkce pro sjednocení délky vzorků (ořezání)
def truncate_signals(data, target_length):
    truncated_data = []
    for signal in data:
        truncated_signal = signal[:target_length]  # Ořezání na cílovou délku
        truncated_data.append(truncated_signal)
    return np.array(truncated_data)


# Načtení všech dat
def load_dataset(input_dir, target_length):
    data = []
    labels = []
    file_names = []  # Ukládáme původní názvy souborů
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            signal = np.loadtxt(file_path)
            label = 1 if "_healthy_" in filename else 0  # Označení tříd (1 = healthy, 0 = unhealthy)
            data.append(signal)
            labels.append(label)
            file_names.append(filename)  # Ukládáme původní název
    # Ořezání délky všech vzorků
    data = truncate_signals(data, target_length)
    return np.array(data), np.array(labels), file_names


# Funkce pro uložení dat zpět do souborů
def save_balanced_dataset(data, labels, file_names, output_dir):
    for i, (signal, label) in enumerate(zip(data, labels)):
        if i < len(file_names):
            # Pokud jde o původní vzorek, zachováme původní název
            original_name = file_names[i]
            output_path = os.path.join(output_dir, original_name)
        else:
            # Pokud jde o syntetický vzorek, vytvoříme nový název
            label_str = "healthy" if label == 1 else "unhealthy"
            output_path = os.path.join(output_dir, f"synthetic_signal_{i}_{label_str}.txt")
        np.savetxt(output_path, signal)
        print(f"Uloženo: {output_path}")


# Parametry
input_dir = "E:\\PyCharmProjects\\voice_procesing_kan\\venv_data_sources\\data_set"
output_dir = "E:\\PyCharmProjects\\voice_procesing_kan\\venv_data_sources\\smote_data_set"
os.makedirs(output_dir, exist_ok=True)

# Nastavení pevné délky vzorků (např. 5000 bodů)
target_length = 19665

# Načtení dat a jejich ořezání
data, labels, file_names = load_dataset(input_dir, target_length)

# Normalizace
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Aplikace SMOTE
smote = SMOTE(sampling_strategy='minority', random_state=0)
data_balanced, labels_balanced = smote.fit_resample(data_normalized, labels)

# Denormalizace zpět do původního měřítka
data_balanced_original = scaler.inverse_transform(data_balanced)

# Uložení nově vytvořených dat
save_balanced_dataset(data_balanced_original, labels_balanced, file_names, output_dir)
