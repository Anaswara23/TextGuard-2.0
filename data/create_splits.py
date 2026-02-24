import pandas as pd
import numpy as np

df = pd.read_csv('chunks_all_docs_labeled_clean.csv')
df = df.dropna(subset=['chunk_text'])
df = df[df['compliance_status'] != 'Not Applicable'].reset_index(drop=True)

# Doc-level label: Non-Compliant if ANY chunk is NC (IDB definition)
label_map = {'Compliant': 0, 'Non-Compliant': 1}
doc_info = (
    df.groupby('document_id')['compliance_status']
    .apply(lambda x: int(x.map(label_map).dropna().max()))
    .reset_index()
    .rename(columns={'compliance_status': 'doc_label'})
)

print("Doc-level label distribution (any-NC aggregation):")
print(doc_info['doc_label'].value_counts().to_string())

# Separate by class
compliant_docs    = doc_info[doc_info['doc_label'] == 0]['document_id'].tolist()
noncompliant_docs = doc_info[doc_info['doc_label'] == 1]['document_id'].tolist()

print(f"\nCompliant docs: {len(compliant_docs)}, Non-Compliant: {len(noncompliant_docs)}")

# Guarantee ≥1 Compliant doc in every split (manually assign)
np.random.seed(42)
np.random.shuffle(compliant_docs)
np.random.shuffle(noncompliant_docs)

# Each split gets exactly 1 Compliant doc
train_docs = [compliant_docs[0]] if len(compliant_docs) > 0 else []
val_docs   = [compliant_docs[1]] if len(compliant_docs) > 1 else []
test_docs  = [compliant_docs[2]] if len(compliant_docs) > 2 else []

# Fill remaining slots with Non-Compliant docs (target: 25 train / 8 val / 7 test)
remaining_nc = noncompliant_docs
n_nc_train   = 24
n_nc_val     = 7
n_nc_test    = len(remaining_nc) - n_nc_train - n_nc_val

train_docs += remaining_nc[:n_nc_train]
val_docs   += remaining_nc[n_nc_train:n_nc_train + n_nc_val]
test_docs  += remaining_nc[n_nc_train + n_nc_val:]

split_map = {d: 'train' for d in train_docs}
split_map.update({d: 'val'  for d in val_docs})
split_map.update({d: 'test' for d in test_docs})
df['split'] = df['document_id'].map(split_map)

df.to_csv('chunks_with_splits.csv', index=False)

print("\n=== Split verification ===")
for split in ['train', 'val', 'test']:
    sub     = df[df['split'] == split]
    doc_sub = doc_info[doc_info['document_id'].isin(sub['document_id'].unique())]
    n_nc    = (doc_sub['doc_label'] == 1).sum()
    n_c     = (doc_sub['doc_label'] == 0).sum()
    print(f"{split}: {sub['document_id'].nunique()} docs  "
          f"[Compliant:{n_c}  Non-Compliant:{n_nc}]  {len(sub)} chunks")