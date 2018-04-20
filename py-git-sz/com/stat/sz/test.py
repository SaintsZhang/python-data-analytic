
import numpy as np
import pandas as pd

#Plotting tools
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sbn
from sklearn_ext.metrics import lorenz
import itertools


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

import gzip
import pickle
#gf = gzip.GzipFile(r"C:\Saints\Data\Script\Python\Saints\solvay-poc\Solvay-POC\data.gz")
#res = gf.read()  # Read the raw content of the file
#result = pickle.loads(res)  # Converts it to a Pandas DataFrame object
result = pd.read_pickle(r"C:\Saints\Data\Script\Python\Saints\solvay-poc\Solvay-POC\data.gz",compression='gzip')

print(result.head(1))
