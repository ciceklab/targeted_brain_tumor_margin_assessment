from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc)


# measure classification performance using confusion matrix, auroc, aupr, precision, recall, f1 score and accuracy
def measure_model_performance(pred, pred_prob, label):
    cm = confusion_matrix(label, pred)
    auroc = roc_auc_score(label, pred_prob)
    p,r,t = precision_recall_curve(label, pred_prob)
    aupr = auc(r,p)
    prec = precision_score(label, pred, average="binary")
    rec = recall_score(label, pred, average="binary")
    f1 = f1_score(label, pred, average="binary")
    acc = accuracy_score(label, pred)
    return cm, auroc, aupr, prec, rec, f1, acc