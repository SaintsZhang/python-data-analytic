#Building predictive model for the Credit Management Proof of Concept

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

sbn.set_style("whitegrid")

def draw_importance(feature_names, feature_scores, ax=None, cmap=None):
    """Graph to visualize feature importances"""
    feat_imp = zip(feature_names, feature_scores)
    feat_imp.sort(key=lambda x: x[1])
    
    fnames, fscores = zip(*feat_imp)
    nfeat = len(fscores)
    
    if ax is None:
        _, ax = plt.subplots(1, 1)

    _, _, patches = ax.hist(
        range(nfeat),
        bins=range(nfeat + 1),
        weights=fscores,
        orientation="horizontal"
    )
    ax.set_yticks(np.linspace(0, nfeat, nfeat+1) + 0.5)
    ax.set_yticklabels(fnames)
    ax.set_ylabel("Feature")
    ax.set_xlabel("score")
    ax.set_ylim(0, nfeat)

    nscores = (fscores - min(fscores)) / (max(fscores) - min(fscores))
    if cmap is not None:
        colors = cmap(np.linspace(0.2, 0.8, len(patches)))
        for p, c in zip(patches, colors):
            p.set_color(c)

def draw_confusion_matrix(y_true, y_pred, labels, ax=None, cmap=None):
    """Draw a confusion matrix"""
    conf_mat = confusion_matrix(y_true, y_pred)

    if ax is None:
        _, ax = plt.subplots(1, 1)
    sbn.heatmap(conf_mat, annot=True, fmt="d", ax=ax, cbar=False, cmap=None)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels[::-1], rotation=0)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Truth")       

def curves(func, xlabel, ylabel, y_true,
           scores, labels="", ax=None,
           xlim=None, ylim=None, score_func=None,
           xfactor=None, yfactor=None, plot_func="plot"):
    """Draw multiple curves related to scoring"""
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores =scores.reshape(1, -1)
        labels = [labels]

    if ax is None:
        _, ax = plt.subplots(1, 1)

    plot_func = getattr(ax, plot_func)
    linestyle = ["-", "--", "-.", ":"]
    for score, label, ls in zip(scores, labels, itertools.cycle(linestyle)):
        label = label
        if score_func is not None:
            label = ". ".join([label, "score: %0.2f" % score_func(y_true, score)])
        x, y = func(y_true, score)[:2]
        if xfactor is not None:
            x = [xval * xfactor for xval in x]
        if yfactor is not None:
            y = [yval * yfactor for yval in y]
        plot_func(x, y, ls, label=label)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best")

def draw_roc_curves(y_true, scores, labels="", ax=None,
                    xlabel="False Positive Rate",
                    ylabel="True Positive Rate", **kwargs):
    """Draw multiple ROC curves"""
    return curves(roc_curve, "False Positive Rate", "True Positive Rate",
                  y_true, scores, labels,
                  ax=ax, score_func=roc_auc_score, **kwargs)
    
def draw_precision_recall_curves(y_true, scores, labels="", xlabel="Recall",
                                 ylabel="Precision", ax=None, **kwargs):
    """Draw multiple precision recall curves"""
    def prec_rec(y_true, score):
        prec, rec = precision_recall_curve(y_true, score)[:2]
        return rec, prec
    return curves(prec_rec, "Recall", "Precision", y_true, scores, labels=labels, ax=ax,
                  score_func=average_precision_score, **kwargs)

def draw_lorenz_curves(amount, scores, labels="", ax=None, xlabel="", ylabel="",
                       score_func=lorenz.lorenz_auc_score, **kwargs):
    """Draw multiple lorenz (lift) curves"""
    return curves(lorenz.lorenz_curve, xlabel, ylabel,
                  amount, scores, labels, ax,
                  score_func=score_func, **kwargs)

def draw_hist(x, y, xlabel="", ylabel="", ax=None, figsize=None, xrotation=0, **hist_opt):
    """Draw bar chart using the histogram function"""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    nbins = len(x)
    ax.hist(range(nbins), range(nbins + 1), weights=y, **hist_opt)
    ax.set_xticks(np.linspace(0, nbins-1, nbins) + 0.5)
    ax.set_xticklabels(x, rotation=xrotation)
    ax.set_xlim(0, nbins)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
#Loading the data

import gzip
import pickle
gf = gzip.GzipFile(r"C:\Saints\Data\Script\Python\Saints\solvay-poc\Solvay-POC\data.gz")
res = gf.read()  # Read the raw content of the file
result = pickle.loads(res)  # Converts it to a Pandas DataFrame object
print(result.columns)

# definition of training and testing sets
from sklearn.utils import check_random_state
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn_ext.preprocessing import feature_transformers as ft
reload(ft)


target_name = "Delay Category"
target = result["Delay Category"].values
y = target > delay_thresh

amount = result["Amount EUR"]
late_amount = amount * y

date_thresh = pd.datetools.to_datetime("2014-12-31").date()
clear_dates = result["Clear Date"]
posting_dates = result["Posting Date"]
itrain = np.where(clear_dates <= date_thresh)[0]
itest = np.where(posting_dates > date_thresh)[0]

#compute the scores based on the payment index
y_pred_pi = result["Payment Index"].iloc[itest] == '200'
y_test = y[itest]
print("With PI")
print(classification_report(y_test, y_pred_pi))

plt.figure()
ax = plt.subplot('111')
draw_confusion_matrix(y_test, y_pred_pi, [r"In time", "Late"], ax=ax)
ax.set_title("Payment Index only")
plt.savefig("Confusion_matrix_payment_index_only.png")

#Feature selection and data transformation
reload(ft)
from sklearn.ensemble import RandomForestClassifier

# We exclude risk features
risk_feats = ["Risk Manual", "Payment Index", "Rating"]

# We exclude features which are irrelevant or which may include
# some forward looking information
base_exclude =  ["Customer CM ID", "Customer ID",
                 "Day of Year",
                 "Risk Category",
                 "Delay", "Sales employee", "Company code",
                 "Dunning Level", "Was dunned", "Dunn blocked", "Scenario",
                 "Rel. Dev Ex. Rate",
                 "Posting Date", "Due Date", "Clear Date", "Last Clear Date",
                 ]

exclude = base_exclude + risk_feats

result_ = result[[col for col in result.columns.values if col not in exclude + [target_name]]]
print (result_.columns)

# Those features are categorical and are preprocessed to be
# usable with sklearn trees
categorical = ["Country", "Cust. Country",
               "Industry", "Industry Code", "Sub Activity",
               "Local currency", "Document currency", 'Geographic Zone',
               'Mini Zone', 'Customer Class']

data = ft.FeatureSetTransformer(
    transformer=ft.FeatureFactorOrderer(target_type="categorical"),
    features=categorical,
    verbose=True
).fit(result_, y).transform(result_)

# Some nan values may appear for customer with no history, we replace them by -1
# so that they are easily separable by the algorithm. An imputation strategy
# may be used instead
data[np.isnan(data)] = -1

# Split the data set
X_train = data[itrain]
X_test = data[itest]
y_train = y[itrain]
y_test = y[itest]

#Train the model and perform prediction
rng = check_random_state(1)
cl = RandomForestClassifier(
    random_state=rng, max_depth=11, criterion="entropy",
    n_estimators=50, n_jobs=6, class_weight='auto'
).fit(X_train, y_train)
y_pred = cl.predict(X_test)
y_pred_proba = cl.predict_proba(X_test)[:, 1]

#Analyse the result
late_amount = result["Amount EUR"] * y
amount = result["Amount EUR"]

print("AUC:", roc_auc_score(y_test, y_pred_proba))
print("AUC lorenz:", lorenz.lorenz_auc_score(late_amount[itest].values,
                                             amount[itest].values * y_pred_proba))
print( classification_report(y_test, y_pred))
_, ax = plt.subplots(1, 1, figsize=(7, 8))
draw_importance(X_test.features, cl.feature_importances_, cmap=cm.BuPu, ax=ax)
plt.subplots_adjust(left=0.35)
plt.savefig("Feature_importance_invoice_features.png")
draw_confusion_matrix(y_test, y_pred, [r"In time", "Late"])
plt.savefig("Confusion_matrix_invoice_features.png", dpi=200)
draw_roc_curves(y_test, y_pred_proba)
draw_precision_recall_curves(y_test, y_pred_proba)

#Reference strategy
#We build some ranking similar to what is done by cash collection teams
reload(lorenz)
sbn.set_palette("Dark2")

amount_cust = result.iloc[itest][["Customer ID", "Amount EUR", "Payment Index", "Delay"]].copy()
amount_cust.index = range(len(amount_cust))
amount_cust["Late amount"] = amount_cust["Amount EUR"] * y_test
amount_cust["Expected late amount"] = amount_cust["Amount EUR"] * y_pred_proba

# Top 5 percents are checked whatever happens
n_top_5 = int(np.round(len(amount_cust) * 0.05))
top_5_inv = amount_cust.sort(columns=["Amount EUR"]).index.values[-n_top_5:]

select_in_top_5_inv = np.zeros(len(amount_cust), "bool")
select_in_top_5_inv[top_5_inv] = True

amount_pi = amount_cust[["Payment Index", "Amount EUR"]].copy()
pi = amount_pi["Payment Index"].copy()
pi[select_in_top_5_inv] = '300'
amount_pi["Payment Index"]=pi
score_pi = np.argsort(amount_pi.sort(columns=["Payment Index", "Amount EUR"]).index.values)

amount = amount_cust["Amount EUR"]
late_amount = amount * y_test

#Figures for final report
#Global curve without result

# Lift (Lorenz) curves for the invoices (without our model)
_, ax = plt.subplots(1, 1, dpi=600)

draw_lorenz_curves(
    late_amount,
    [amount,
     score_pi,
     late_amount],
     labels=["Amount",
            "Payment Index + 5%",
            "Ideal"],
    xlabel="Percentage of invoices", ylabel=u"Overdue in M¢ã",
    xfactor=100, yfactor=late_amount.sum() / 1000000.,
    ax=ax, score_func=None
)


# Draw the gray area betweeen the curves
xval1, yval1 = lorenz.lorenz_curve(late_amount, amount)
xval2, yval2 = lorenz.lorenz_curve(late_amount, score_pi)

ax.fill_between(
    xval1 * 100,
    yval1 * late_amount.sum() / 1e6,
    yval2 * late_amount.sum() / 1e6, facecolor='lightgray')

plt.savefig("report_curves_init_state.png", dpi=200)

#Focus on 30% of the lowest 95%
# Zoom on lowest 95%
_, ax = plt.subplots(1, 1)

xval1, yval1 = lorenz.lorenz_curve(late_amount[~select_in_top_5_inv], amount[~select_in_top_5_inv])
xval2, yval2 = lorenz.lorenz_curve(late_amount[~select_in_top_5_inv], score_pi[~select_in_top_5_inv])

ax.fill_between(
    xval1 * 100,
    yval1 * late_amount[~select_in_top_5_inv].sum() / 1e6,
    yval2 * late_amount[~select_in_top_5_inv].sum() / 1e6, facecolor='lightgray')


draw_lorenz_curves(
    late_amount[~select_in_top_5_inv],
    [amount[~select_in_top_5_inv],
     score_pi[~select_in_top_5_inv],
     late_amount[~select_in_top_5_inv]],
    labels=[
            "Amount",
            "Payment Index",
            "Ideal"],
    xlabel="Percentage of invoices", ylabel=u"Overdue in M¢ã",
    xlim=(0, 30), ylim=(0, 600), xfactor=100, yfactor=late_amount[~select_in_top_5_inv].sum() / 1000000.,
    ax=ax, score_func=None
)

ax.set_title("Focus on Lowest 95%")
plt.savefig("report_curves_init_state_zoom.png", dpi=200)

#Global curve with the result
# Lift (Lorenz) curves for the invoices with our model

_, ax = plt.subplots(1, 1)


# Draw the gray area
xval1, yval1 = lorenz.lorenz_curve(late_amount, amount)
xval2, yval2 = lorenz.lorenz_curve(late_amount, score_pi)

ax.fill_between(
    xval1 * 100,
    yval1 * late_amount.sum() / 1e6,
    yval2 * late_amount.sum() / 1e6, facecolor='lightgray')



# We keep the result curve separate 
# so we can choose the way it's displayed
xval3, yval3 = lorenz.lorenz_curve(
    late_amount,
    amount_cust["Expected late amount"])

# Draw all curves except the result
draw_lorenz_curves(
    late_amount,
    [amount,
     score_pi,
     late_amount],
    labels=[
            "Amount",
            "Payment Index",
            "Ideal"],
    xlabel="Percentage of invoices", ylabel=u"Overdue in M¢ã",
    xfactor=100, yfactor=late_amount.sum() / 1000000.,
    ax=ax, score_func=None
)

ax.plot(xval3 * 100, yval3 * late_amount.sum() / 1e6, r'-', lw=2,
        label="Machine learning")
ax.legend(loc="best")

plt.savefig("report_curves_ml.png", dpi=200)

# Focus on lowest 95%
_, ax = plt.subplots(1, 1)

xval1, yval1 = lorenz.lorenz_curve(late_amount[~select_in_top_5_inv], amount[~select_in_top_5_inv])
xval2, yval2 = lorenz.lorenz_curve(late_amount[~select_in_top_5_inv], score_pi[~select_in_top_5_inv])

ax.fill_between(
    xval1 * 100,
    yval1 * late_amount[~select_in_top_5_inv].sum() / 1e6,
    yval2 * late_amount[~select_in_top_5_inv].sum() / 1e6, facecolor='lightgray')


xval3, yval3  = lorenz.lorenz_curve(
    late_amount[~select_in_top_5_inv],
    amount_cust["Expected late amount"][~select_in_top_5_inv])



draw_lorenz_curves(
    late_amount[~select_in_top_5_inv],
    [amount[~select_in_top_5_inv],
     score_pi[~select_in_top_5_inv],
     late_amount[~select_in_top_5_inv]],
    labels=[
            "Amount",
            "Payment Index",
            "Ideal"],
    xlabel="Percentage of invoices", ylabel=u"Overdue in M¢ã",
    xlim=(0, 30), ylim=(0, 600), xfactor=100, yfactor=late_amount[~select_in_top_5_inv].sum() / 1000000.,
    ax=ax, score_func=None
)

ax.plot(xval3 * 100, yval3 * late_amount[~select_in_top_5_inv].sum() / 1e6, r'-', lw=2,
        label="Machine learning")
ax.legend(loc="best")
ax.set_title("Focus on Lowest 95%")
plt.savefig("report_curves_ml_zoom.png", dpi=200)

# Compute improvement at 30%
i_30 = np.argmin(np.abs(xval1 - 0.3))
delta = (yval3 - np.maximum(yval1, yval2))[i_30]
print("For the lowest 95% invoices")
print(u"Improvement: %0.1f M¢ã" % (delta * late_amount[~select_in_top_5_inv].sum() / 1e6))
print("Relative improvement: %0.1f %%" % (delta / np.maximum(yval1, yval2)[i_30] * 100.))

amounts_95 = amount_cust[~select_in_top_5_inv].copy()
n_30 = int(np.round(len(amounts_95) * 0.3))
top_30 = amounts_95.sort(columns=["Expected late amount"]).iloc[-n_30:]
cust_30 = top_30.groupby("Customer ID").sum().sort(columns=["Expected late amount"], ascending=False)
#print cust_30
cust_30.to_csv("Top_30_percent_customers.csv", index=False)

# Compute average delay weighted by amounts
print
print("Agverages on the full database")
delays = result["Delay"][y].copy().values
amounts = result["Amount EUR"][y].copy().values

print("Weighted average: %0.1f days" % ((delays * amounts).sum() / amounts.sum()))
print("Unweighted average: %0.1f days" % delays.mean())

#Compute average on the retrieved 30%

# Compute average delay weighted by amounts
print
print("Averages on the top 30% retrieved")
delays30 = top_30["Delay"]
amounts30 = top_30["Amount EUR"][delays30 > 0]
delays30 = top_30["Delay"][delays30 > 0]
print("Weighted average: %0.1f days" % ((delays30 * amounts30).sum() / amounts30.sum()))
print("Unweighted average: %0.1f days" % delays30.mean())

result_q = result.copy()
result_q["target"] = (result["Delay Category"] > 0).astype(int)

gb = result_q[["Company code", "target"]].groupby("Company code")

vals = (gb.sum() / gb.count()).reset_index()
vals = list(vals.to_records(index=False))
vals.sort(key=lambda x: -x[-1])

cc, val = zip(*vals)
draw_hist(cc[:100], val[:100], "Company Code", "Proportion of late invoices", figsize=(20, 6), xrotation=70)
plt.savefig("figures/hist_company_codes_late_rate.svg")

gb = result_q[["Sales employee", "target"]].groupby("Sales employee")
vals = (gb.sum() / gb.count()).reset_index()
vals = list(vals.to_records(index=False))
vals.sort(key=lambda x: -x[-1])

se, val = zip(*vals)
draw_hist(se[:100], val[:100], "Sales Employee", "Proportion of late invoices", figsize=(20, 6), xrotation=70)
plt.savefig("figures/hist_sales_employee_late_rate.svg")
    