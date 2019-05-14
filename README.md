# Matthews Correlation Coefficient (MCC)
MCC function for ML

Here I would like to share my implementation of Matthews Correlation Coefficient (MCC) for various situations.

*Inspiration from [Kaggle kernel by Michal](https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric)* on "Best loss function for F1-score metric".

## Intro on MCC

I encountered MCC while search for the ["best multi-class classification metric"](https://sebastianraschka.com/faq/docs/multiclass-metric.html).

[Wikipedia](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient#Multiclass_case) has very nice explanation of MCC, while at [stats.stackexchange](https://stats.stackexchange.com/questions/187768/matthews-correlation-coefficient-with-multi-class) you can find a very interesting discussion on the topic. Multi-class MCC is often called "R_K statistics" so I found [the whole page](http://rk.kvl.dk/introduction/index.html) devoted to it.

The most useful for computation was Eq.(8) from the original article by [Gorodkin](https://www.sciencedirect.com/science/article/pii/S1476927104000799?via%3Dihub):

<a href="https://www.codecogs.com/eqnedit.php?latex=R_K&space;=&space;\frac{N\,&space;\mathrm{Tr}(C)&space;-&space;\sum_{kl}&space;\tilde{C_k}&space;\hat{C_l}}{\sqrt{N^2&space;-&space;\sum_{kl}&space;\tilde{C_k}&space;\hat{C^T_l}}\sqrt{N^2&space;-&space;\sum_{kl}&space;\tilde{C^T_k}&space;\hat{C_l}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R_K&space;=&space;\frac{N\,&space;\mathrm{Tr}(C)&space;-&space;\sum_{kl}&space;\tilde{C_k}&space;\hat{C_l}}{\sqrt{N^2&space;-&space;\sum_{kl}&space;\tilde{C_k}&space;\hat{C^T_l}}\sqrt{N^2&space;-&space;\sum_{kl}&space;\tilde{C^T_k}&space;\hat{C_l}}}" title="R_K = \frac{N\, \mathrm{Tr}(C) - \sum_{kl} \tilde{C_k} \hat{C_l}}{\sqrt{N^2 - \sum_{kl} \tilde{C_k} \hat{C^T_l}}\sqrt{N^2 - \sum_{kl} \tilde{C^T_k} \hat{C_l}}}" /></a>

where N is the number of examples, \tilde{C_k} is the k*th* row of the confusion matrix C, \hat{C_l} the l*th* column of C, C^T is C transposed and Tr(C) is the trace of C.

Note: if you only need MCC value to be computed use [sklearn.metrics.matthews_corrcoef](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)!

## binary_mcc_loss.py

Function that can be used as loss function for Keras training in the binary classification case.

## multi_mcc_loss.py

Function that can be used as loss function for Keras training in the multi-class classification case.

## plot_mcc_vs_tresh.py

Following that one can  precision/recall vs tresholds, I wanted to see how MCC behaves.
