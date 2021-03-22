import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

dataset = pd.read_excel('../Dengue_Dataset.xlsx') 


#cross-validation
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values
# smaller model
def create_smaller():
	# create model
	model = Sequential()
	model.add(Dense(26, input_dim=26,kernel_regularizer='l1', activation='relu'))
	model.add(Dense(30,input_dim=26,kernel_regularizer='l2',activation='relu'))
	model.add(Dense(30,input_dim=30,kernel_regularizer='l2',activation='relu'))
	
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
	return model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=120, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))