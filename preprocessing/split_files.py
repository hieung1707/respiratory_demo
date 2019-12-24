import glob
import random
import re


patient_ids = set()
audio_txts = glob.glob('/home/hieung1707/projects/respiratory_demo/data/audio_and_txt_files/*.txt')
for txt in audio_txts:
    filename = txt.split('/')[-1]
    patient_id = filename.split('.')[0].split('_')[0]
    patient_ids.add(int(patient_id))

patient_ids = list(patient_ids)
print(len(patient_ids))

val_split = 0.1
test_split = 0.1
training_files = []
test_files = []
val_files = []

selected_indices = random.sample(range(0, len(patient_ids)),  int(val_split*len(patient_ids)))

# list_keys = list(wav.keys())
selected_ids = [patient_ids[idx] for idx in selected_indices]
print(len(selected_ids))