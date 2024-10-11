import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rc('font',family='Times New Roman')

f, (ax0, ax1) = plt.subplots(2, 1)
f.set_figwidth(8)
f.set_figheight(12)


label_issues_before = pd.read_csv('/Users/project/CL/clmethods/label_issues_M2.csv')
label_issues_before['ML0'] = np.where(label_issues_before['given_label']== 0, label_issues_before['label_quality'], 1-label_issues_before['label_quality'])
label_issues_before['ML1'] = np.where(label_issues_before['given_label']== 1, label_issues_before['label_quality'], 1-label_issues_before['label_quality'])


label_issues_afters1 = pd.read_csv('/Users/project/CL/clmethods/label_issues_M2_afters1.csv')
label_issues_afters1['ML0'] = np.where(label_issues_afters1['given_label']== 0, label_issues_afters1['label_quality'], 1-label_issues_afters1['label_quality'])
label_issues_afters1['ML1'] = np.where(label_issues_afters1['given_label']== 1, label_issues_afters1['label_quality'], 1-label_issues_afters1['label_quality'])

label_issues_afters2 = pd.read_csv('/Users/project/CL/clmethods/label_issues_M2_afters2.csv')
label_issues_afters2['ML0'] = np.where(label_issues_afters2['given_label']== 0, label_issues_afters2['label_quality'], 1-label_issues_afters2['label_quality'])
label_issues_afters2['ML1'] = np.where(label_issues_afters2['given_label']== 1, label_issues_afters2['label_quality'], 1-label_issues_afters2['label_quality'])


numbins = 25
a_heights, a_bins = np.histogram(label_issues_before.loc[label_issues_before['given_label']==0]['ML0'], bins=numbins)
b_heights, b_bins = np.histogram(label_issues_before.loc[label_issues_before['given_label']==1]['ML1'], bins=numbins)
c_heights, c_bins = np.histogram(label_issues_afters1.loc[label_issues_afters1['given_label']==0]['ML0'], bins=numbins)
d_heights, d_bins = np.histogram(label_issues_afters1.loc[label_issues_afters1['given_label']==1]['ML1'], bins=numbins)
e_heights, e_bins = np.histogram(label_issues_afters2.loc[label_issues_afters2['given_label']==0]['ML0'], bins=numbins)
f_heights, f_bins = np.histogram(label_issues_afters2.loc[label_issues_afters2['given_label']==1]['ML1'], bins=numbins)

a_heights = a_heights/len(label_issues_before.loc[label_issues_before['given_label']==0])
b_heights = b_heights/len(label_issues_before.loc[label_issues_before['given_label']==1])
c_heights = c_heights/len(label_issues_afters1.loc[label_issues_afters1['given_label']==0])
d_heights = d_heights/len(label_issues_afters1.loc[label_issues_afters1['given_label']==1])
e_heights = e_heights/len(label_issues_afters2.loc[label_issues_afters2['given_label']==0])
f_heights = f_heights/len(label_issues_afters2.loc[label_issues_afters2['given_label']==1])

ax0.plot(a_bins[:-1], a_heights, color='#94A9D8', label = 'l=0 Samples before Pruning ')
ax0.plot(c_bins[:-1], c_heights, color='#385492', ls = '--', label = 'l=0 Samples after Pruning RightBetterThanWrongConfident')
ax0.plot(e_bins[:-1], e_heights, color='#385492', ls = '-.', label = 'l=0 Samples after Pruning WrongBetterButNotConfident')


ax1.plot(b_bins[:-1], b_heights, color='#94A9D8',  label = 'l=1 Samples before Pruning')
ax1.plot(d_bins[:-1], d_heights, color='#385492', ls = '--', label = 'l=1 Samples after Pruning RightBetterThanWrongConfident')
ax1.plot(f_bins[:-1], f_heights, color='#385492', ls = '-.', label = 'l=1 Samples after Pruning WrongBetterButNotConfident')


ax0.legend(loc="upper left",fontsize=13)
ax1.legend(loc="upper left",fontsize=13)

ax0.set_xlabel('(a) Predicted Probability (l = 0)',fontsize=18)
ax1.set_xlabel('(b) Predicted Probability (l = 1)',fontsize=18)
ax0.set_ylabel('Sample Proportions',fontsize=14)
ax1.set_ylabel('Sample Proportions',fontsize=14)
ax0.set_ylim((0,0.10))
ax1.set_ylim((0,0.14))

plt.show()


f.savefig('plotcurve.pdf',dpi=1000,format='pdf')

