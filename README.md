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

---

## 📸 Dashboard Preview
![Dashboard](screenshots/dashboard.png) <img width="1440" height="821" alt="dashboard" src="https://github.com/user-attachments/assets/f1daca50-40fe-40d9-9cd9-80f05358828c" />


![Graph](screenshots/graph.png)

![Real Data Test](screenshots/real_data_test.png)


---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python generate_data.py
streamlit run app.py
