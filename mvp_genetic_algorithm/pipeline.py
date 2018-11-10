import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('data.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.9857142857142858
exported_pipeline = make_pipeline(
    StandardScaler(),
    GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features=0.6000000000000001, min_samples_leaf=4, min_samples_split=11, n_estimators=100, subsample=0.9000000000000001)
)


# tf = [[72,35.4,30.4,9.0,20.1,44.9,3.7,10.0,36.7,8.7,10.1,85.8,0.6,4.8,5.4,8.8,1.8,0.7,4.4,30.1]] 	# James Harding
# tf = [[82,36.9,27.5,10.5,19.3,54.2,1.8,5.0,36.7,4.7,6.5,73.1,1.2,7.5,8.6,9.1,1.4,0.9,4.2,32.7]] 	# LeBron James
tf = [[75,36.4,28.1,10.4,19.5,53.4,0.7,2.2,34.0,6.6,8.0,82.8,2.5,8.6,11.1,2.3,1.5,2.6,2.2,33.0]] 	# Anthony Davis


exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(tf)

print(results)


# 79,33.4,20.5,7.5,17.2,43.7,2.4,7.0,34.0,3.0,3.8,80.5,0.7,3.1,3.7,3.7,1.5,0.3,2.7,16.6