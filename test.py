import pickle

subject_ids = set()
with open('data/test_files.pkl', 'rb') as pkl_file:
    subjects = pickle.load(pkl_file)
    pkl_file.close()

for subject in subjects:
    subject_id = int(subject.split('_')[0])
    subject_ids.add(subject_id)

subject_ids = list(subject_ids)
print(subject_ids[:11])
print(len(subject_ids))