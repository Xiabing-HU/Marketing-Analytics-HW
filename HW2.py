import pandas as pd
from sklearn import mixture
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime


data = pd.read_csv('transactions_n100000.csv')
data = data.iloc[:,0:5]

data.order_timestamp = pd.to_datetime(data['order_timestamp'])
order_timestamp2 = pd.to_datetime(data['order_timestamp'])
# data['order_timestamp'] = [d.date() for d in data.order_timestamp]
# data['time'] = [d.time() for d in order_timestamp2]
data = data.drop(columns=['order_timestamp'])


# OneHotEncoder
location = data.location.astype('category')
df_loc = pd.DataFrame(location)
# Assigning numerical values and storing in another column
df_loc['location_cat'] = df_loc['location'].cat.codes
# creating instance of one-hot-encoder
enc = OneHotEncoder(categories='auto', handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df = pd.DataFrame(enc.fit_transform(df_loc[['location_cat']]).toarray())
# merge with main df bridge_df on key values
df_loc = df_loc.join(enc_df)
df_loc = df_loc.drop(columns=['location'])


# Dummy
item_name = data.item_name.astype('category')
df_item = pd.DataFrame(item_name)
# generate binary values using get_dummies
dum_df = pd.get_dummies(df_item, columns=["item_name"], prefix=["Type_is"] )
# merge with main df bridge_df on key values
df_item = df_item.join(dum_df)
df_item = df_item.drop(columns=['item_name'])

data_new = data.join(df_item)
data_new = data_new.join(df_loc)
data_new.set_index(['ticket_id'], inplace=True)
print(data_new)


feature_x = [tag for tag in data_new.columns if tag not in ['location_cat','location','item_name']]
print('feature:\n', feature_x)
X = data_new[feature_x].values
#X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# How many clusters?
n_components = np.arange(1, 11)
models = [mixture.GaussianMixture(n, covariance_type='full', random_state=1350).fit(X) for n in n_components]
plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
#plt.plot(n_components, [m.aic(X_train) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.show()

# GMM
GMM = mixture.GaussianMixture(n_components=8, covariance_type='full')
gmm = GMM.fit(X)
labels = gmm.predict(X)
data_new['cluster'] = labels
data_new.to_csv('clustered.csv')