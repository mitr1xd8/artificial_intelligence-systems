import numpy as np
from sklearn import preprocessing

input_data = np.array([[-1.3, 3.9, 4.5], [-5.3, -4.2,-1.3], [5.2, -6.5, -1.1], [-5.2, 2.6, -2.2]])
data_binarized = preprocessing.Binarizer(threshold=3.0).transform(input_data)
print("\n Binarized data:\n", data_binarized)
data_scaled = preprocessing.scale(input_data)
print("\nAFTER: ")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nМin max scaled data:\n", data_scaled_minmax)
data_normalized_l1 = preprocessing.normalize(input_data,
norm='l1')
print("\nl1 normalized data:\n", data_normalized_l1)

