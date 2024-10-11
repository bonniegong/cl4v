import pandas as pd
import numpy as np


label_issues = pd.read_csv('/Users/project/CL/clmethods/label_issues_M2.csv')
MLW = label_issues.loc[label_issues['is_label_issue']==True]

train_sample = pd.read_parquet('/Users/project/CL/train_qemu.parquet')
train_sample = train_sample.loc[~train_sample['newidx'].isin(MLW['newidx'])]
train_sample.to_csv('/Users/project/CL/pruneandtest/train_qemu_M2_remove.csv')
print(8000-len(train_sample))

train_sample = pd.read_parquet('/Users/project/CL/train_qemu.parquet')
false_neg = MLW.loc[MLW['given_label']==0]
false_pos = MLW.loc[MLW['given_label']==1]
train_sample.loc[train_sample['newidx'].isin(false_neg['newidx']), 'label'] = 1
train_sample.loc[train_sample['newidx'].isin(false_pos['newidx']), 'label'] = 0
train_sample.to_csv('/Users/project/CL/pruneandtest/train_qemu_M2_invert.csv')
print(8000-len(train_sample))

t0 = label_issues.loc[label_issues['given_label']==0]['label_quality'].mean()
t1 = label_issues.loc[label_issues['given_label']==1]['label_quality'].mean()
label_issues['ML0'] = np.where(label_issues['given_label']== 0, label_issues['label_quality'], 1-label_issues['label_quality'])
label_issues['ML1'] = np.where(label_issues['given_label']== 1, label_issues['label_quality'], 1-label_issues['label_quality'])

rmv1 = label_issues.loc[((t0 > label_issues['ML0']) & (label_issues['ML0'] > label_issues['ML1']) & (label_issues['ML1'] > t1) & (label_issues['given_label']==0)) | ((t1 > label_issues['ML1']) & (label_issues['ML1'] > label_issues['ML0']) & (label_issues['ML0'] > t0) & (label_issues['given_label']==1))]
rmv2 = label_issues.loc[((t0 > label_issues['ML0']) & (label_issues['ML0'] > label_issues['ML1']) & (label_issues['given_label']==1)) | ((t1 > label_issues['ML1']) & (label_issues['ML1'] > label_issues['ML0'])& (label_issues['given_label']==0))]

train_sample = pd.read_parquet('/Users/project/CL/train_qemu.parquet')
train_sample = train_sample.loc[~train_sample['newidx'].isin(rmv1['newidx'])]
train_sample.to_csv('/Users/project/CL/pruneandtest/train_qemu_s1.csv')
print(8000-len(train_sample))

train_sample = pd.read_parquet('/Users/project/CL/train_qemu.parquet')
train_sample = train_sample.loc[~train_sample['newidx'].isin(rmv2['newidx'])]
train_sample.to_csv('/Users/project/CL/pruneandtest/train_qemu_s2.csv')
print(8000-len(train_sample))

