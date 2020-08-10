import uvicorn
from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pickle
import numpy as np


model = pickle.load(open('finalized_model.sav', 'rb'))


# init app
app = FastAPI()


# Routes
@app.get('/')
async def index():
	return {"text": "To use the micro service use '/predict/query' the query should be the variables"
					" separated by ; like in the dataset"}


def modify_inputs(query):
	query = query.replace('"', '')
	query = query.replace('.', '')
	query = query.replace(',', '.')
	query = query.replace('NA;', '')
	query = query.replace('NA', '')

	inputs = query.split(';')
	inputs[1] = float(inputs[1])
	inputs[2] = float(inputs[2])
	inputs[7] = float(inputs[7])
	inputs[10] = int(inputs[10])
	inputs[13] = float(inputs[13])
	inputs[13] = int(inputs[13])
	inputs[14] = float(inputs[14])
	inputs[15] = int(inputs[15])
	data = [
		[inputs[0], inputs[2], inputs[3], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9], inputs[10], inputs[11],
		 inputs[12], inputs[13]]]
	columns = ['variable1', 'variable3', 'variable4', 'variable6', 'variable7', 'variable8', 'variable9', 'variable10',
			   'variable11', 'variable12', 'variable13', 'variable14']
	X_test = pd.DataFrame(data, columns=columns)

	labelencoder = LabelEncoder()
	X_test['variable1'] = labelencoder.fit_transform(X_test['variable1'])
	X_test['variable9'] = labelencoder.fit_transform(X_test['variable9'])
	X_test['variable10'] = labelencoder.fit_transform(X_test['variable10'])
	X_test['variable12'] = labelencoder.fit_transform(X_test['variable12'])

	Nomials = ['variable4', 'variable6', 'variable7', 'variable13']
	X_test = pd.get_dummies(X_test, columns=Nomials, prefix=Nomials)

	org = ['variable1', 'variable3', 'variable8', 'variable9', 'variable10', 'variable11', 'variable12', 'variable14',
		   'variable4_u', 'variable4_y', 'variable6_W', 'variable6_aa', 'variable6_c', 'variable6_cc', 'variable6_d',
		   'variable6_e', 'variable6_ff', 'variable6_i', 'variable6_j', 'variable6_k', 'variable6_m', 'variable6_q',
		   'variable6_x', 'variable7_bb', 'variable7_dd', 'variable7_ff', 'variable7_h', 'variable7_j', 'variable7_n',
		   'variable7_v', 'variable7_z', 'variable13_g', 'variable13_s']
	exist = X_test.columns.values
	diff = [x for x in org if x not in exist]
	temp = pd.DataFrame(np.zeros((1, len(diff))))
	temp.columns = diff
	print(len(org), org)
	print(len(exist), exist)
	print(len(diff), diff)
	result = pd.concat([X_test, temp], axis=1)
	result = result[org]
	return result

# ML Aspect
@app.get('/predict/{query}')
async def predict(query):
	y_pred = model.predict(modify_inputs(query))
	result = 0
	if y_pred[0] == 1:
		result = 'yes'

	return {"For the query": query, "The classification is": result}



if __name__ == '__main__':
	uvicorn.run(app,host="127.0.0.1",port=8000)