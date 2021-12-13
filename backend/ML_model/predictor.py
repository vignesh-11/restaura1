from main import *

dp = r'Test.csv'
rd = pd.read_csv('Dataset/Test.csv')

rd2 = pd.read_csv('Dataset/zomato.csv')
rd3 = rd2.append(rd, ignore_index = True)

tr1, nr1 = common_pipeline.fit_transform(rd3)

nr_prep = full_pipeline.fit_transform(nr1.drop('target', axis=1))
print(nr_prep.shape)

model = pickle.load(open('ml_model.pkl','rb'))

y_pd = model.predict(nr_prep)
y_pb = model.predict_proba(nr_prep)
y_sc = y_pb[:, 1]

# Labelling new data
nr1['success_class'] = y_pd
nr1['success_proba'] = y_sc
nr1['success_proba'].iloc[-1]

nrd = nr1.reset_index().merge(rd3.reset_index()[['name', 'index']], how='left', on='index')
#print(nrd.iloc[-1])
val = nrd.iloc[-1]["success_proba"]
val = 100 * val
print(val)
