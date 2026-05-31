# 🚀 Anomaly Detection in Oil & Gas Sensor Data

This project builds a machine learning system to detect anomalies in industrial sensor data such as pressure, temperature, flow rate, and vibration.

---

## 🔧 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn (Isolation Forest)
- Streamlit
- Plotly

---

## 📊 Features
- Detects abnormal sensor behavior using machine learning  
- Interactive dashboard for visualization  
- Supports multiple sensor types  
- Highlights anomalies in real time  

## Key Results
- Built a machine learning pipeline to identify abnormal patterns in industrial sensor data.
- Applied the Isolation Forest algorithm to detect anomalous observations.
- Processed and visualized sensor readings including pressure, temperature, flow rate, and vibration.
- Developed an interactive dashboard using Streamlit and Plotly for real-time anomaly monitoring.
---

## 📸 Dashboard Preview
![Dashboard](screenshots/dashboard.png) <img width="1440" height="821" alt="dashboard" src="https://github.com/user-attachments/assets/f1daca50-40fe-40d9-9cd9-80f05358828c" />


---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python generate_data.py
streamlit run app.py
