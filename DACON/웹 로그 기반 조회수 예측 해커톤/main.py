import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

X_train = train_data.drop(['TARGET', 'sessionID', 'userID'], axis=1)
y_train = train_data['TARGET']

label_encoder = LabelEncoder()
for column in X_train.columns:
    if X_train[column].dtype == 'object':
        combined_data = pd.concat([X_train[column], test_data[column]], axis=0)
        label_encoder.fit(combined_data)
        X_train[column] = label_encoder.transform(X_train[column])
        test_data[column] = label_encoder.transform(test_data[column])

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

test_predictions = model.predict(test_data.drop(['sessionID', 'userID'], axis=1))

submission = pd.DataFrame({'sessionID': test_data['sessionID'], 'TARGET': test_predictions})

submission.to_csv('submission.csv', index=False)