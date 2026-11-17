import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("dataset/indian_house_prices_dataset.csv")

plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)

sns.barplot(data=df, x="City", y="Price_INR_Lakhs")

plt.title("House Prices by City")
plt.ylabel("Price (INR Lakhs)")
plt.xticks(rotation=45, ha="right")


plt.subplot(2, 2, 2)

sns.scatterplot(
    data=df,
    x="Super_Area_SqFt",
    y="Price_INR_Lakhs"
)

plt.title("House Area vs Price")
plt.xlabel("Super Area (SqFt)")
plt.ylabel("Price (INR Lakhs)")

plt.subplot(2, 2, 3)
sns.histplot(data=df,x="Price_INR_Lakhs")
plt.xlabel("House Price (INR Lakhs)")
plt.ylabel("Frequency")


plt.tight_layout(pad=3)
plt.savefig("house_prices.png", dpi=300, bbox_inches="tight")
plt.show()