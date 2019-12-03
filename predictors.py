def log_reg_regularized(X_train, Y_train, X_val):
    from sklearn.linear_model import LogisticRegression
    from statsmodels.tools import add_constant
    X_train = add_constant(X_train)
    X_val = add_constant(X_val)
    lr = LogisticRegression(fit_intercept = False)
    lr.fit(X_train, Y_train)
    Y_pred_prob = lr.predict_proba(X_val)[:,1]
    return Y_pred_prob
    
def log_reg(X_train, Y_train, X_val):
    from statsmodels.discrete.discrete_model import Logit
    from statsmodels.tools import add_constant
    X_train = add_constant(X_train)
    X_val = add_constant(X_val)
    logit = Logit(Y_train, X_train)
    fit = logit.fit(method = 'bfgs', maxiter = 10000)
    logitprobs = fit.predict(X_val)  
    return logitprobs

def NN(X_train, Y_train, X_val, Y_val, model, epochs = 30):
    model.fit(X_train, Y_train, epochs = epochs, validation_data = (X_val, Y_val))
    prob = model.predict(X_val)
    return prob