from flask import Flask, render_template, request, send_file, redirect, session
import pickle
import numpy as np
from reportlab.pdfgen import canvas
import sqlite3
from rapidfuzz import process
app = Flask(__name__)
app.secret_key = "ai_health_secret"
# ---------- DATABASE SETUP ----------
conn = sqlite3.connect("history.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
    disease TEXT,
    probability REAL
)
""")
# User table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT
)
""")
conn.commit()
# Load ML Model
model = pickle.load(open("model.pkl", "rb"))
symptoms_list = pickle.load(open("symptoms.pkl", "rb"))

# Precaution Dictionary
precautions_dict = {
    "Fungal infection": "Keep skin clean and dry.",
    "Allergy": "Avoid allergens and take antihistamines.",
    "Bronchial Asthma": "Avoid dust & smoke, carry inhaler.",
    "Diabetes": "Control sugar level, exercise daily.",
    "Flu": "Take rest, drink fluids, take paracetamol."
}

# Doctor Recommendation
doctor_dict = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist",
    "Bronchial Asthma": "Pulmonologist",
    "Diabetes": "Endocrinologist",
    "Flu": "General Physician"
}

# ---------- Prediction Function ----------
def predict_disease(selected_symptoms):

    input_vector = [0] * len(symptoms_list)

    for symptom in selected_symptoms:
        if symptom in symptoms_list:
            index = symptoms_list.index(symptom)
            input_vector[index] = 1

    prediction = model.predict([input_vector])[0]
    probability = max(model.predict_proba([input_vector])[0]) * 100

    precaution = precautions_dict.get(prediction, "Consult doctor")
    doctor = doctor_dict.get(prediction, "General Physician")
    return prediction, round(probability, 2), precaution, doctor

# Doctor Recommendation
doctor_dict = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist",
    "Bronchial Asthma": "Pulmonologist",
    "Diabetes": "Endocrinologist",
    "Flu": "General Physician"
}

# ---------- PDF FUNCTION ----------
def generate_pdf(prediction, probability):
    c = canvas.Canvas("report.pdf")
    c.drawString(100, 750, "AI Medical Report")
    c.drawString(100, 700, f"Disease: {prediction}")
    c.drawString(100, 650, f"Probability: {probability}%")
    c.save()
# ---------- Routes ----------
@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")
    
    return render_template("index.html", symptoms=symptoms_list)


@app.route("/predict", methods=["POST"])
def predict():

    selected_symptoms = request.form.getlist("symptoms")

    input_vector = [0] * len(symptoms_list)

    for symptom in selected_symptoms:
        if symptom in symptoms_list:
            index = symptoms_list.index(symptom)
            input_vector[index] = 1

    # ---- Probability Prediction ----
    probabilities = model.predict_proba([input_vector])[0]

    disease_names = model.classes_

    # Combine disease + probability
    disease_prob_list = list(zip(disease_names, probabilities))

    # Sort top 3 diseases
    top3 = sorted(disease_prob_list, key=lambda x: x[1], reverse=True)[:5]

    # Main prediction
    prediction = top3[0][0]
    probability = round(top3[0][1] * 100, 2)
    cursor.execute(
    "INSERT INTO history (disease, probability) VALUES (?, ?)",
    (prediction, probability)
    )
    conn.commit()
    generate_pdf(prediction, probability)
    precaution = precautions_dict.get(prediction, "Consult doctor")
    doctor = doctor_dict.get(prediction, "General Physician")
    map_link = "https://www.google.com/maps/search/" + doctor
    return render_template(
        "index.html",
        symptoms=symptoms_list,
        prediction=prediction,
        probability=probability,
        precaution=precaution,
        doctor=doctor,
        top3=top3,
        map_link=map_link

    )



@app.route("/ml_chat", methods=["POST"])
def ml_chat():

    data = request.get_json()
    user_msg = data.get("message", "").lower()

    # Match symptoms using fuzzy logic
    detected_symptoms = []

    for symptom in symptoms_list:
        match_score = process.extractOne(symptom, [user_msg])[1]
        if match_score > 70:   # threshold
            detected_symptoms.append(symptom)

    if len(detected_symptoms) == 0:
        reply = "Please describe symptoms like fever, cough, headache etc."
        return {"reply": reply}

    prediction, probability, precaution, doctor = predict_disease(detected_symptoms)

    reply = f"""
    Possible Disease: {prediction}
    Confidence: {probability}%
    Precaution: {precaution}
    Doctor: {doctor}
    """

    return {"reply": reply}

@app.route("/download_report")
def download_report():
    return send_file("report.pdf", as_attachment=True)

@app.route("/history")
def history():
     if "user" not in session:
          return redirect("/login")
     
     cursor.execute("SELECT * FROM history")
     data = cursor.fetchall()
     return render_template("history.html", data=data)

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )

        user = cursor.fetchone()

        if user:
            session["user"] = username
            return redirect("/")
        else:
            return render_template("login.html", error="Invalid Login")

    return render_template("login.html")
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")
@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        try:
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            conn.commit()

            return redirect("/login")

        except:
            return render_template("signup.html", error="Username already exists")

    return render_template("signup.html")
if __name__ == "__main__":
    app.run(debug=True)
