import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("testfile.csv")
df["Date"] = pd.to_datetime(df["Date"])
#-----------------------------------------------------------------
plt.figure(figsize=(16, 10))
plt.plot(df["Date"], df["AbsoluteTemp"], color="darkblue", label="Absolute Temperature (°C)")
plt.title("Monthly Absolute Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#-----------------------------------------------------------------
# 기준: 33도 이상
heatwave_df = df[df["AbsoluteTemp"] >= 33]
non_heatwave_df = df[df["AbsoluteTemp"] < 33]

plt.figure(figsize=(16, 5))

plt.plot(df["Date"], df["AbsoluteTemp"], color="skyblue", label="Absolute Temperature")

plt.scatter(heatwave_df["Date"], heatwave_df["AbsoluteTemp"], color="red", label="Heatwave (≥33°C)", s=10)

plt.title("Monthly Absolute Temperature with Heatwave Markers")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.axhline(33, color="gray", linestyle="--", linewidth=1, label="Heatwave Threshold (33°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#---------------------------------------------------------------
df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["dayofyear"] = df["Date"].dt.dayofyear
df["season"] = df["month"] % 12 // 3 + 1  # 1:겨울, 2:봄, 3:여름, 4:가을

df["temp_lag1"] = df["AbsoluteTemp"].shift(1)  # 전월 온도
df["temp_lag2"] = df["AbsoluteTemp"].shift(2)
df["temp_diff"] = df["AbsoluteTemp"] - df["temp_lag1"]  # 전월 대비 온도 변화

df["temp_roll3"] = df["AbsoluteTemp"].rolling(window=3).mean()  # 3개월 이동평균
df["temp_roll5"] = df["AbsoluteTemp"].rolling(window=5).mean()

df["heatwave_streak"] = (df["is_heatwave"].rolling(window=2).sum() == 2).astype(int)

df = df.dropna().reset_index(drop=True)

df.head()
#-------------------------------------------------------------------------
features = ["month", "season", "temp_lag1", "temp_lag2", "temp_diff", "temp_roll3", "temp_roll5", "heatwave_streak"]
X = df[features]
y = df["is_heatwave"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Heatwave", "Heatwave"], yticklabels=["No Heatwave", "Heatwave"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
