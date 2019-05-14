import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as prc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def plot_mcc_vs_tresh(y_true, y_pred):
    ### Getting thresholds:
    _, _, thresholds = prc(y_true, y_pred)
    ### Getting all prediction with predefined thresholds:
    all_pred_tt = [np.where(y_pred > tt, 1, 0) for tt in thresholds]
    ### One-Hot encoding is needed for faster Confusion Matrix Calculation
    ohe = OneHotEncoder(categories='auto', sparse=False)
    ohe.fit(y_true.reshape(-1, 1))
    ### Confusion Matrix = Y_true^T * Y_Pred
    all_cm = [np.matmul(np.transpose(ohe.transform(y_true.reshape(-1, 1))), ohe.transform(all_pred_tt[tt].reshape(-1, 1))) for tt in range(len(thresholds))]
    ### MultiMCC Nominator:
    all_up = [all_cm[tt][1, 1]*all_cm[tt][0, 0] - all_cm[tt][0, 1]*all_cm[tt][1, 0] for tt in range(len(thresholds))]
    ### MultiMCC Denominator:
    all_down = [np.sqrt((all_cm[tt][1, 1]+all_cm[tt][0, 1]) * (all_cm[tt][1, 1]+all_cm[tt][1, 0]) * (all_cm[tt][0, 0]+all_cm[tt][0, 1]) * (all_cm[tt][0, 0]+all_cm[tt][1, 0])) for tt in range(len(thresholds))]   
    ### Final calculation:
    mcc = [x/y for x, y in zip(all_up, all_down)]
    ### Plotting
    plt.plot(thresholds, matthew_cc, 'b--', label='MatthewsCC')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
#     plt.ylim([-1,1]) ### those are theoretical minimum and maximum for MCC
