import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from saints.python.funcs import Funcs

sat_17 = pd.read_csv('./data/sat_2017.csv')
sat_18 = pd.read_csv('./data/sat_2018.csv')
act_17 = pd.read_csv('./data/act_2017.csv')
act_18 = pd.read_csv('./data/act_2018.csv')

sat_17.name = 'SAT 2017'
sat_18.name = 'SAT 2018'
act_17.name = 'ACT 2017'
act_18.name = 'ACT 2018'

df_list = [sat_17, sat_18, act_17, act_18]

#Funcs.print_props(df_list, '.dtypes')

act_17['Composite'] = act_17['Composite'].apply(lambda x_cell: x_cell.strip('x'))
#print(act_17['Composite'].value_counts(ascending = False))
#print(Funcs.compare_values(act_18['State'], sat_18['State']))
#print(act_18[act_18['State'] == 'National'])
act_18.drop(act_18.index[23], inplace = True)
act_18['State'].replace({'Washington, D.C.':'District of Columbia'}, inplace = True)
act_18.sort_values(by=['State'], inplace=True)
sat_18.sort_values(by=['State'], inplace=True)
sat_18 = sat_18.reset_index(drop=True)
act_18 = act_18.reset_index(drop=True)

act_17['Participation'] = act_17['Participation'].apply(lambda cells: cells.strip('%'))
act_18['Participation'] = act_18['Participation'].apply(lambda cells: cells.strip('%'))

sat_17['Participation'] = sat_17['Participation'].apply(lambda cells: cells.strip('%'))
sat_18['Participation'] = sat_18['Participation'].apply(lambda cells: cells.strip('%'))

act_17 = Funcs.convert_to_float(act_17)
sat_17 = Funcs.convert_to_float(sat_17)
act_18 = Funcs.convert_to_float(act_18)
sat_18 = Funcs.convert_to_float(sat_18)

act_17.drop(act_17.index[0], inplace=True)
act_17 = act_17.reset_index(drop=True)

act_18.drop(act_18.index[19], inplace=True)
act_18 = act_18.reset_index(drop=True)

new_act_17_cols = {
    'State':'state',
    'Participation':'act_participation_17',
    'Composite':'act_composite_17'
}
act_17.rename(columns=new_act_17_cols, inplace=True)
act_17.name = 'ACT 2017'

sat_17.drop(columns = ['Evidence-Based Reading and Writing', 'Math'], inplace = True)

# rename the 2017 SAT columns
new_sat_17_cols = {
    'State':'state',
    'Participation':'sat_participation_17',
    'Total':'sat_score_17'
    }
sat_17.rename(columns=new_sat_17_cols, inplace=True)
sat_17.name = 'SAT 2017'

df_list = [sat_17, act_17]

act_17.drop(columns = ['English', 'Math', 'Reading', 'Science'], inplace = True)
sat_act_17 = pd.merge(sat_17, act_17, left_index=True, on = 'state', how='outer')
sat_act_17.name = 'SAT/ACT 2017'

# rename the 2018 ACT columns
new_act_18_cols = {
    'State':'state',
    'Participation':'act_participation_18',
    'Composite':'act_composite_18'
}
act_18.rename(columns=new_act_18_cols, inplace=True)
act_18.name = 'ACT 2018'

# rename the 2018 SAT columns
new_sat_18_cols = {
    'State':'state',
    'Participation':'sat_participation_18',
    'Total':'sat_score_18'
    }
sat_18.rename(columns=new_sat_18_cols, inplace=True)
sat_18.name = 'SAT 2018'

df_list = [act_18, sat_18]
sat_18.drop(columns = ['Evidence-Based Reading and Writing', 'Math'], inplace=True)

sat_act_18 = pd.merge(sat_18, act_18, left_index=True, on = 'state', how='outer')
sat_act_18 = sat_act_18.reset_index(drop=True)
sat_act_18.name = 'SAT/ACT 2018'

df_list = [sat_act_17, sat_act_18]

sat_act_17.to_csv('./data/sat_act_17.csv', encoding='utf-8')
sat_act_18.to_csv('./data/sat_act_18.csv', encoding='utf-8')

sat_act = pd.merge(sat_act_17, sat_act_18, left_index=True, on = 'state', how='outer')
sat_act.to_csv('./data/sat_act_2017_2018.csv', encoding='utf-8')

df = pd.read_csv('./data/sat_act_2017_2018.csv')
df.drop(columns = ['Unnamed: 0'], axis=1, inplace = True)

plt.figure(figsize = (15,10))
plt.title('SAT and ACT Correlation Heatmap', fontsize = 16);

# Mask to remove redundancy from the heatmap.
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax = 1, cmap = "coolwarm",  annot = True);

plt.figure(figsize = (8,6))
features = ['sat_participation_17', 'sat_participation_18', 'act_participation_17', 'act_participation_18']
plt.title('SAT and ACT Participation Rate Correlations', fontsize = 16);
mask = np.zeros_like(df[features].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df[features].corr(), mask=mask, vmin=-1, vmax = 1, cmap = "coolwarm",  annot = True);

plt.figure(figsize = (8,6))
features = ['sat_score_17', 'sat_score_18', 'act_composite_17', 'act_composite_18']
plt.title('Average SAT Score vs Average ACT Composite Score Correlations', fontsize = 16);
mask = np.zeros_like(df[features].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df[features].corr(), mask=mask, vmin=-1, vmax = 1, cmap = "coolwarm",  annot = True);

fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (15,12))

sns.boxplot(df.sat_participation_17, ax = ax[0,0], orient="h", color = 'orange').set(
    xlabel='', title='SAT Participation Rates 2017');

sns.boxplot(df.sat_participation_18, ax = ax[0,1], orient="h", color = 'orange').set(
    xlabel='', title='SAT Participation Rates 2018');

sns.boxplot(df.act_participation_17, ax = ax[1,0], orient="h", color= 'pink').set(
    xlabel='', title='ACT Participation Rates 2017');

sns.boxplot(df.act_participation_18, ax = ax[1,1], orient="h", color = 'pink').set(
    xlabel='', title='ACT Participation Rates 2018');

plt.tight_layout();

plt.show()

