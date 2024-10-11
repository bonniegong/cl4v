import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rc('font',family='Times New Roman')
label_issues = pd.read_csv('/Users/project/CL/clmethods/label_issues_M2.csv')


label_issues['ML0'] = np.where(label_issues['given_label']== 0, label_issues['label_quality'], 1-label_issues['label_quality'])
label_issues['ML1'] = np.where(label_issues['given_label']== 1, label_issues['label_quality'], 1-label_issues['label_quality'])


MLW0 = label_issues.loc[(label_issues['is_label_issue']==True) & (label_issues['given_label']==0)]
MLW1 = label_issues.loc[(label_issues['is_label_issue']==True) & (label_issues['given_label']==1)]

t0 = label_issues.loc[label_issues['given_label']==0]['label_quality'].mean()
t1 = label_issues.loc[label_issues['given_label']==1]['label_quality'].mean()

label_issues0 = label_issues.loc[label_issues['given_label']==0]
label_issues1 = label_issues.loc[label_issues['given_label']==1]


rmv1_0 = label_issues.loc[(t0 > label_issues['ML0']) & (label_issues['ML0'] > label_issues['ML1']) & (label_issues['ML1'] > t1) & (label_issues['given_label']==0)]
rmv1_1 = label_issues.loc[(t1 > label_issues['ML1']) & (label_issues['ML1'] > label_issues['ML0']) & (label_issues['ML0'] > t0) & (label_issues['given_label']==1)]

rmv2_0 = label_issues.loc[(t1 > label_issues['ML1']) & (label_issues['ML1'] > label_issues['ML0']) & (label_issues['given_label']==0)]
rmv2_1 = label_issues.loc[(t0 > label_issues['ML0']) & (label_issues['ML0'] > label_issues['ML1']) & (label_issues['given_label']==1)]

good0 = label_issues0.loc[ (~label_issues0['newidx'].isin(MLW0['newidx'])) & (~label_issues0['newidx'].isin(rmv1_0['newidx'])) & (~label_issues0['newidx'].isin(rmv2_0['newidx']))]
good1 = label_issues1.loc[ (~label_issues1['newidx'].isin(MLW1['newidx'])) & (~label_issues1['newidx'].isin(rmv1_1['newidx'])) & (~label_issues1['newidx'].isin(rmv2_1['newidx']))]

f, (ax0, ax1) = plt.subplots(1, 2)

binRange = np.arange(0.0,1.0,0.02)
heights_y0, a_bins0 = np.histogram(good0['ML0'], bins=binRange)
heights_y0_ml, b_bins0 = np.histogram(MLW0['ML0'], bins=binRange)
heights_y0_rm1, c_bins0 = np.histogram(rmv1_0['ML0'], bins=binRange)
heights_y0_rm2, d_bins0 = np.histogram(rmv2_0['ML0'], bins=binRange)

heights_y1, a_bins1 = np.histogram(good1['ML1'], bins=binRange)
heights_y1_ml, b_bins1 = np.histogram(MLW1['ML1'], bins=binRange)
heights_y1_rm1, c_bins1 = np.histogram(rmv1_1['ML1'], bins=binRange)
heights_y1_rm2, d_bins1 = np.histogram(rmv2_1['ML1'], bins=binRange)

width = (a_bins1[1] - a_bins1[0])/1.5

ax0.bar(a_bins0[:-1], heights_y0, width=width, facecolor='#DCE3F2', label='Samples with no Issue')
ax0.bar(b_bins0[:-1], heights_y0_ml, width=width, facecolor='#94A9D8', label = 'Mislabeled Samples')
ax0.bar(d_bins0[:-1], heights_y0_rm2, width=width, facecolor='#385492', label = 'WrongBetterButNotConfident Samples')
ax0.grid(True)
ax0.set_xlabel('(a) Predicted Probability (l = 0)',fontsize=24)
ax0.set_ylabel('Sample Numbers',fontsize=18)

ax1.bar(a_bins1[:-1], heights_y1, width=width, facecolor='#DCE3F2', label='Samples with no Issue')
ax1.bar(b_bins1[:-1], heights_y1_ml, width=width, facecolor='#94A9D8', label = 'Mislabeled Samples')
ax1.bar(c_bins1[:-1], heights_y1_rm1, width=width, facecolor='#253761', label = 'RightBetterThanWrongConfident Samples')

ax1.grid(True)
ax1.set_xlabel('(b) Predicted Probability (l = 1)',fontsize=24)
ax0.legend(loc="upper left",fontsize=15)
ax1.legend(loc="upper left",fontsize=15)
ax0.set_ylim((0,200))
ax1.set_ylim((0,200))

f.set_figwidth(14)
f.set_figheight(6)
plt.show()

f.savefig('plot12.pdf',dpi=1000,format='pdf')

