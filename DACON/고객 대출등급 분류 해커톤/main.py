import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder  # 추가

# 데이터 로드
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
submission_data = pd.read_csv("sample_submission.csv")

# 데이터 전처리
X_train = train_data.drop(['ID', '대출등급'], axis=1)
y_train = train_data['대출등급']

X_test = test_data.drop('ID', axis=1)

def extract_term(x):
    if 'year' in str(x):
        return int(x.split()[0]) * 12
    elif 'month' in str(x):
        return int(x.split()[0])
    else:
        try:
            return int(x)
        except ValueError:
            return int(''.join(filter(str.isdigit, str(x))))

X_train['대출기간'] = X_train['대출기간'].apply(extract_term)
X_test['대출기간'] = X_test['대출기간'].apply(extract_term)

X_train['근로기간'] = X_train['근로기간'].str.extract('(\d+)').astype(float)
X_test['근로기간'] = X_test['근로기간'].str.extract('(\d+)').astype(float)

X_train['연간소득'] = X_train['연간소득'].replace(',', '', regex=True).astype(float)
X_test['연간소득'] = X_test['연간소득'].replace(',', '', regex=True).astype(float)

X_train['대출금액'] = X_train['대출금액'].replace(',', '', regex=True).astype(float)
X_test['대출금액'] = X_test['대출금액'].replace(',', '', regex=True).astype(float)

X_train['대출기간'] = pd.to_numeric(X_train['대출기간'], errors='coerce')
X_test['대출기간'] = pd.to_numeric(X_test['대출기간'], errors='coerce')

X_train['대출금액_대출기간_곱'] = X_train['대출금액'] * X_train['대출기간']
X_test['대출금액_대출기간_곱'] = X_test['대출금액'] * X_test['대출기간']

X_train['부채_대비_소득'] = X_train['부채_대비_소득_비율'] * X_train['연간소득']
X_test['부채_대비_소득'] = X_test['부채_대비_소득_비율'] * X_test['연간소득']


X_train = pd.get_dummies(X_train, columns=['주택소유상태', '대출목적'])
X_test = pd.get_dummies(X_test, columns=['주택소유상태', '대출목적'])

missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0

X_test = X_test[X_train.columns]

numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='most_frequent')
X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

# 대출등급을 Label Encoding으로 변환
label_encoder = LabelEncoder()  # 추가
y_train_encoded = label_encoder.fit_transform(y_train)  # 추가

# 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train_encoded)  # y_train_encoded로 변경

# 테스트 데이터로 예측
test_preds = model.predict(X_test)

# 예측 결과를 sample_submission.csv 형식으로 저장
submission_df = pd.DataFrame({'ID': submission_data['ID'], '대출등급': label_encoder.inverse_transform(test_preds)})  # 추가
submission_df.to_csv('submission.csv', index=False)
