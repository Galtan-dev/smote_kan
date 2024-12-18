import os
import random
import pandas as pd
import opensmile

random.seed(0)

def data_from_txt(path):
    with open(path, 'r') as file:
        data = [float(line.strip()) for line in file]
    return data

# Cesty ke složkám a souborům
csv_path = "E:\\PyCharmProjects\\voice_procesing_kan\\venv_data_sources\\file_information.csv"
data_path = "E:\\PyCharmProjects\\voice_procesing_kan\\venv_data_sources\\data_set"
augmented_path = "E:\\PyCharmProjects\\voice_procesing_kan\\venv_data_sources\\augmented_records"
smote_path = "E:\\PyCharmProjects\\voice_procesing_kan\\venv_data_sources\\smote_data_set"
txt_features_output = "E:\\PyCharmProjects\\voice_procesing_kan\\smote_implementation\\features_output.txt"
txt_status_output = "E:\\PyCharmProjects\\voice_procesing_kan\\smote_implementation\\status_output.txt"

# Parametry
sample_rate = 50000
augment_selection = {"noise": 0, "shifted": 0, "speed": 100}  # Počet souborů pro každou augmentaci, 0 znamená žádný výběr
smote_count = 100  # Počet "synthetic_signal" souborů

# Načtení informací o unikátních a duplicitních vzorcích
data = pd.read_csv(csv_path, header=None, skiprows=1)
duplicate_samples = data[data.duplicated(subset=[4], keep=False)]
unique = data.drop(duplicate_samples.index)
unique_indexes = unique[0].tolist()

# Inicializace OpenSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals
)

# Funkce pro výběr náhodných souborů
def select_random_files(path, prefix, count, filter_func=None):
    files = [f for f in os.listdir(path) if f.startswith(prefix) and (filter_func(f) if filter_func else True)]
    return random.sample(files, min(count, len(files)))

# Zpracování a ukládání výsledků
status_info = []
healthy_count = 0
unhealthy_count = 0

with open(txt_features_output, "w") as features_file, open(txt_status_output, "w") as status_file:
    # Zpracování hlavního datasetu
    for idx, item in enumerate(os.listdir(data_path)):
        try:
            file_index = int(item.split("svdadult")[1].split("_")[0])
            status = item.split("_")[1].split("_")[0]
            if file_index in unique_indexes:
                data_bank = data_from_txt(os.path.join(data_path, item))
                features = smile.process_signal(data_bank, sample_rate).values.flatten().tolist()
                features_file.write(",".join(map(str, features)) + "\n")
                status_file.write(status + "\n")

                # Počítání statusů
                if status == "healthy":
                    healthy_count += 1
                elif status == "unhealthy":
                    unhealthy_count += 1

        except Exception as ex:
            print(f"Error processing file {item}: {ex}")
    print("Data_set processed")

    # Zpracování augmentovaných záznamů
    for augment, count in augment_selection.items():
        if count > 0:
            selected_files = select_random_files(augmented_path, "", count, lambda f: augment in f)
            for item in selected_files:
                try:
                    data_bank = data_from_txt(os.path.join(augmented_path, item))
                    features = smile.process_signal(data_bank, sample_rate).values.flatten().tolist()
                    features_file.write(",".join(map(str, features)) + "\n")
                    status_file.write(augment + "\n")
                except Exception as ex:
                    print(f"Error processing file {item}: {ex}")
    print("Augmentations processed")

    # Zpracování SMOTE záznamů
    smote_files = select_random_files(smote_path, "synthetic_signal", smote_count, lambda f: "healthy" in f)
    for item in smote_files:
        try:
            data_bank = data_from_txt(os.path.join(smote_path, item))
            features = smile.process_signal(data_bank, sample_rate).values.flatten().tolist()
            features_file.write(",".join(map(str, features)) + "\n")
            status_file.write("synthetic" + "\n")
        except Exception as ex:
            print(f"Error processing file {item}: {ex}")
    print("Smote processed")

# Výpis výsledků
print(healthy_count)
print(f"Total healthy files: {healthy_count + smote_count + sum(augment_selection.values())}")
print(f"Total unhealthy files: {unhealthy_count}")
