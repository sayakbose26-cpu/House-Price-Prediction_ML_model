import joblib
import pandas as pd

model = joblib.load("house_price_model.pk1")

# Collect inputs
city = input("Enter city (e.g., Mumbai, Delhi, Bangalore, Pune, Chennai, Ahmedabad): ")
locality_type = input("Enter locality type (Affordable / Mid-Range / Premium): ")
super_area = float(input("Enter super area (sqft): "))
age_of_property = int(input("Enter age of property (years): "))
distance_city_center = float(input("Enter distance to city center (km): "))
bathrooms = int(input("Enter the number of bathrooms: "))
bhk = int(input("Enter the BHK type: "))
distance_to_metro = float(input("Enter the distance to metro (km): "))

new_house = pd.DataFrame({
    "City": [city],
    "Locality_Type": [locality_type],
    "Super_Area_SqFt": [super_area],
    "Age_of_Property": [age_of_property],
    "Distance_to_City_Center_km": [distance_city_center],
    "Bathrooms": [bathrooms],
    "BHK": [bhk],
    "Distance_to_Metro_km": [distance_to_metro]
})

predict_y = model.predict(new_house)

print("\n-------------------------------------")
print(f"Predicted house price: {predict_y[0]:.2f} Lakhs")
print("-------------------------------------")