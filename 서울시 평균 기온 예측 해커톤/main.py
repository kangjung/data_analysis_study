import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
sample = pd.read_csv("sample_submission.csv")

df[['최고기온', '최저기온', '일교차', '강수량', '평균습도', '평균풍속', '일조합', '일사합', '일조율']] = df[['최고기온', '최저기온', '일교차', '강수량', '평균습도', '평균풍속', '일조합', '일사합', '일조율']].ffill()
df['강수량'] = df['강수량'].fillna(0)
df['년'] = df['일시'].str[:4]
df['월'] = df['일시'].str[5:7]

df = df.reset_index(drop=True)

features = df.iloc[21185:].drop(['일시', '평균기온'], axis=1)
target = df.iloc[21185:]['평균기온']

features['요일'] = pd.to_datetime(df['일시']).dt.dayofweek
features['일년'] = pd.to_datetime(df['일시']).dt.dayofyear
features['주말'] = features['요일'].isin([5, 6]).astype(int)

cols = features.select_dtypes(include="O").columns
for col in cols:
    features[col] = features[col].astype('category')

test_sizes = [0.1, 0.2, 0.3, 0.4]

best_mae = float('inf')
best_test_size = None

for test_size in test_sizes:
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=test_size, random_state=0)
    lgb_model = LGBMRegressor(random_state=0, n_estimators=1000, learning_rate=0.05, max_depth=10, num_leaves=31)
    lgb_model.fit(X_train, y_train, categorical_feature=list(cols), eval_set=[(X_val, y_val)], eval_metric="mae")
    pred_val = lgb_model.predict(X_val)
    mae_lgb = mean_absolute_error(y_val, pred_val)
    print(f"Test Size: {test_size}, LightGBM MAE: {mae_lgb}")

    if mae_lgb < best_mae:
        best_mae = mae_lgb
        best_test_size = test_size

print(f"Best Test Size: {best_test_size}")

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=best_test_size, random_state=0)
lgb_model = LGBMRegressor(random_state=0, n_estimators=1000, learning_rate=0.05, max_depth=10, num_leaves=31)
lgb_model.fit(X_train, y_train, categorical_feature=list(cols), eval_set=[(X_val, y_val)], eval_metric="mae")

test = df.iloc[22646:].reset_index(drop=True).copy()
test['요일'] = pd.to_datetime(test['일시']).dt.dayofweek
test['일년'] = pd.to_datetime(test['일시']).dt.dayofyear
test['주말'] = test['요일'].isin([5, 6]).astype(int)
test[cols] = test[cols].astype('category')
test_features = test[features.columns]

pred_submit = lgb_model.predict(test_features)

pred_submit = pred_submit[:358]

submission = pd.DataFrame({
    "일시": sample['일시'][:len(pred_submit)],
    "평균기온": pred_submit
})

submission.to_csv('submission_9.csv', index=False)