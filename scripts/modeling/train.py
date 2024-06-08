from sklearn.model_selection import train_test_split

X = preprocessed_fraud_detection_data.drop('Assessment_Binary', axis=1)
y = preprocessed_fraud_detection_data['Assessment_Binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Decision Trees
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Gradient Boosting Machines (GBM)
gbm_model = GradientBoostingClassifier()
gbm_model.fit(X_train, y_train)