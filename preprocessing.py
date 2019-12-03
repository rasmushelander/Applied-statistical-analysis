def preprocess(data_frame, train_val_split = 0.8, output_col = -1):
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
    import numpy as np
    
    # Putting the output variable in the last column 
    cols = data_frame.columns.tolist()
    tempcol = cols[-1]
    cols[-1] = cols[output_col]
    cols[output_col] = tempcol
    data_frame = data_frame[cols]
    
    # One hot encoding the categorical variables 
    categoricals  = data_frame.select_dtypes(include=[object])
    le = LabelEncoder()
    categoricals = categoricals.apply(le.fit_transform)
    enc = OneHotEncoder()
    onehotlabels = enc.fit_transform(categoricals).toarray()

    # Standardising int variables
    ints = data_frame.select_dtypes(include = ['number'])
    scaler = StandardScaler()
    scaler.fit(ints)
    standardized = scaler.transform(ints)

    # Putting all variables in one array
    data = np.concatenate((standardized, onehotlabels), axis = 1)

    # Separating inputs (x) and outputs (y)
    X = np.copy(data[:,:-2])
    Y = np.copy(data[:,-1])
    
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)

    # Separating training and validation sets 
    split = int(len(X)*train_val_split)
    X_train = X [0:split]
    Y_train = Y[0:split]
    X_val = X[split:]
    Y_val =Y[split:]
    
    return(X_train, Y_train, X_val, Y_val)