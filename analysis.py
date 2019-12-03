def metrics(Y_val, predprob, cutoff = 0.5):
    from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
    Y_pred = predictions(Y_val, predprob, cutoff = 0.5)
    d = {}
    d['confusion_matrix'] = confusion_matrix(Y_val, Y_pred)
    d['p'] = precision_score(Y_val, Y_pred)
    d['r'] = recall_score(Y_val, Y_pred)
    d['f1'] = f1_score(Y_val, Y_pred)
    d['acc'] = accuracy_score(Y_val, Y_pred)
    roc = roc_curve(Y_val, predprob)
    d['roc'] = roc
    d['auc'] = auc(roc[0], roc[1])
    
    return d

def plot_roc(res):
    from matplotlib.pyplot import plot as plt 
    r1 = res['roc'][0]
    r2 = res['roc'][1]
    plt.plot(r1,r2)
    return

def predictions(Y_val, predprob, cutoff = 0.5):
    Y_pred = [1 if prob > cutoff else 0 for prob in predprob]
    return Y_pred

def combined_probs(probs, accs, sample_size):
    probs = [prob.reshape((sample_size,)) for prob in probs]
    res = 1/sum(accs)
    a =sum([probs[n]*accs[n] for n in range(len(accs))])
    a = res*a
    return a

    