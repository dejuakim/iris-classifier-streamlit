from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 2. 간단한 모델 학습 (Random Forest)
model = RandomForestClassifier()
model.fit(X, y)

# 3. 모델 저장하기 (이게 핵심!)
joblib.dump(model, 'iris_model.pkl')

print("모델 저장 완료: iris_model.pkl")