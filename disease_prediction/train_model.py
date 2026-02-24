import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train = pd.read_csv("dataset/Training.csv")

train = train.loc[:, ~train.columns.str.contains("^Unnamed")]

X = train.drop("prognosis", axis=1)
y = train["prognosis"]
symptoms_list = X.columns.tolist()

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODELS =================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
gb = GradientBoostingClassifier(random_state=42)

# Ensemble Voting Model
model = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb)],
    voting="soft"
)

# ================= TRAIN MODEL =================
model.fit(X_train, y_train)

# ================= EVALUATION =================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
error = 1 - accuracy

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# ================= SAVE MODEL =================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(symptoms_list, open("symptoms.pkl", "wb"))

print("Model and symptoms saved successfully!")

# ================= CREATE ACCURACY GRAPH =================
plt.figure(figsize=(6, 5))

labels = ["Accuracy", "Error"]
values = [accuracy * 100, error * 100]
colors = ["#4CAF50", "#FF5252"]  # Green & Red

bars = plt.bar(labels, values, color=colors)

plt.ylim(0, 100)
plt.title("Disease Prediction Model Performance")
plt.ylabel("Percentage")

# Show percentage on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.2f}%",
        ha="center",
        fontsize=11
    )

plt.grid(axis="y", linestyle="--", alpha=0.6)

# Save graph
plt.savefig("static/accuracy.png")
plt.close()

print("Accuracy graph saved in static folder!")
