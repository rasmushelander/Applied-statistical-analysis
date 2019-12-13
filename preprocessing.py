class DataHandler():
    
    def __init__(self):
        self.le = None
        self.enc = None
        import numpy as np
        return 
    
    def preprocess(self,data_frame, train_val_split = 0.8, output_col = -1, standardize = True):
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
        import numpy as np
        self.data_frame = data_frame

        # Putting the output variable in the last column 
        cols = data_frame.columns.tolist()
        tempcol = cols[-1]
        cols[-1] = cols[output_col]
        cols[output_col] = tempcol
        data_frame = data_frame[cols]

        # One hot encoding the categorical variables 
        categoricals  = data_frame.select_dtypes(include=[object])
        #self.le = LabelEncoder()
        #categoricals = categoricals.apply(self.le.fit_transform)
        self.enc = OneHotEncoder(handle_unknown = 'ignore')
        onehotlabels = self.enc.fit_transform(categoricals).toarray()

        # Int variables
        ints = data_frame.select_dtypes(include = ['number'])
        intfeatures = len(ints.columns)
        
        # Putting all variables in one array
        data = np.concatenate((ints, onehotlabels), axis = 1)

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
        
        # Standardizing continuous features 
        if standardize:
            scaler = StandardScaler()
            scaler.fit(X_train[:][0:intfeatures])
            X_train[:][0:intfeatures] = scaler.transform(X_train[:][0:intfeatures])
            scaler.fit(X_val[:][0:intfeatures])
            X_val[:][0:intfeatures] = scaler.transform(X_val[:][0:intfeatures])

        return(X_train, Y_train, X_val, Y_val)
    
    def decode_features(self):
        import numpy as np
        ints = self.data_frame.select_dtypes(include = ['number'])
        cols = ints.columns.tolist()
        
        categoricals = self.data_frame.select_dtypes(include = [object])
        onehotlabels = self.enc.fit_transform(categoricals).toarray()
        catlabels = []
        catlabels = self.enc.categories_
        for l in catlabels:
            cols = cols + list(l)
 
        return cols[0:-2]
    
        
        
        
        
        
        