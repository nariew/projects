import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import OrdinalEncoder

# read the data file
df=pd.read_csv("winequalityN.csv")
df.dropna(how='any',inplace=True)

# encoder transform
class type_encoder(object):
	def fit(self,dataFrame):
		self.encoder=OrdinalEncoder(categories='auto')
		self.encoder.fit(np.array(dataFrame.type).reshape(-1,1))

	def transform(self,dataFrame):
		dataFrame['type']=self.encoder.transform(np.array(dataFrame.type).reshape(-1,1))
		return dataFrame

	def fit_transform(self,dataFrame):
		self.fit(dataFrame)
		return self.transform(dataFrame)

# transfer to new label
df=type_encoder().fit_transform(df)

# build the quality encoder
class quality_encoder(object):
	def fit(self,df):
		self.dict={3:0,4:0,5:1,6:1,7:1,8:2,9:2}

	def transform(self,df):
		df['quality']=df.quality.map(self.dict)
		return df

	def fit_transform(self,df):
		self.fit(df)
		return self.transform(df)

# initialize the sub unit of the each component in the final pipeline
smt=BorderlineSMOTE(random_state=2021)
qe=quality_encoder()
xgBoost=xgb.XGBClassifier(
	use_label_encoder=False,gamma=0,reg_alpha=1.5,max_depth=3,eval_metric='mlogloss'
)

# Do the data transfer
df=df[df.type==1]
df['quality']=df.quality.map({3:0,4:0,5:0,6:0,7:1,8:1,9:1})
features=list(df.columns)
target='quality'
features.remove('type')
features.remove(target)
X=df[features]
y=df[target]

# generate the Class for the pipeline
class Pipeline(object):
	def fit(self, X,y=None):
		self.smt=BorderlineSMOTE(random_state=2021)
		self.smt.fit(X,y)
		X_resample,y_resample=self.smt.fit_resample(X,y)
		self.model=xgb.XGBClassifier(
			use_label_encoder=False,gamma=0,reg_alpha=1.5,max_depth=3,eval_metric='mlogloss'
		)
		self.model.fit(X_resample,y_resample)



	def predict_proba(self, X,y=None):
		return self.model.predict_proba(X)

	def MoveThreshHold(self,probArray,thresh_num):
		res=[]
		for eachProba in probArray:
			if eachProba[1]>=thresh_num:
				res.append(1)
			else:
				res.append(0)

		return np.array(res)

	def predict(self, X,y=None):
		return self.MoveThreshHold(self.model.predict_proba(X),0.57)

# Train the model
model=Pipeline()
model.fit(X,y)

# dump the model to be a pkl file
with open('model.pkl', 'wb') as f:
	pickle.dump(model, f)
