import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib 
df = pd.read_csv("dataset/indian_house_prices_dataset.csv")
X=df.drop(columns=["Property_ID","Price_INR_Lakhs","Property_Type","Floor_Number",
    "Total_Floors",
    "Furnishing_Status",
    "Parking",
    "Lift_Available",
    "Gated_Community",
    "Carpet_Area_SqFt"
    ])
y=df["Price_INR_Lakhs"]
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
numerical_feature = X_train.select_dtypes(include = ["int64","float64"]).columns
categorical_feature = X_train.select_dtypes(include=["object"]).columns
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num",numerical_pipeline,numerical_feature),
    ("cat",categorical_pipeline,categorical_feature)
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
model.fit(X_train,y_train)
print("model train completed")
joblib.dump(model,"house_price_model.pk1")

