# 🌌 Night Lights × Stock Returns  
### FIN 377 – Nighttime Satellite Data, Local Economic Activity, and Equity Return Prediction

This repository contains the full research pipeline and Streamlit analytics dashboard for our FIN 377 project, where we examine whether **changes in nighttime brightness around a firm’s headquarters predict its next-month stock returns**.

The project demonstrates data engineering, econometric modeling, geospatial processing, and dashboard development — using **VIIRS nighttime lights**, **HQ geolocation**, **county-level mapping**, and **S&P 500 stock returns**.

---

## 📌 Research Question  
**Do changes in local nighttime brightness around a firm’s headquarters predict its next-month stock return?**

Night-time lights proxy local economic activity — factories, distribution centers, commercial activity, and population dynamics that may signal local economic conditions.

We test whether **ΔLight (month-over-month brightness surprise)** contains **incremental predictive power** for **next-month returns**, after controlling for broad market and seasonal effects.

---

## 📊 Data Sources  

### **1. VIIRS Nighttime Lights (2013–2024)**  
We use the *Visible Infrared Imaging Radiometer Suite* (VIIRS) **Day/Night Band (DNB)** monthly composite dataset.  
The dataset was sourced from:

**Jiaxiong Yao – VIIRS Nighttime Lights Data Index**  
🌐 https://sites.google.com/site/jiaxiongyao16/nighttime-lights-data  

The underlying VIIRS DNB composites were originally produced by:  
**Earth Observation Group (EOG)**, Payne Institute for Public Policy, Colorado School of Mines.

📄 **Citation:**  
Elvidge et al. (2017). *VIIRS Nighttime Lights*. Earth Observation Group, Payne Institute.

🔗 Specific CSV used in this project (Yao’s hosted version):  
https://www.dropbox.com/scl/fi/dxmu3q12hf7ovs0cdmnuz/VIIRS-nighttime-lights-2013m1to2024m5-level2.csv?dl=0  

---

### **2. S&P 500 Firm Headquarters**
Company HQ coordinates were sourced via OpenStreetMap’s Nominatim service.  
Coordinates → County mapping performed using US Census county shapefiles.

---

### **3. S&P 500 Monthly Returns (Yahoo Finance)**  
Monthly stock price data & returns downloaded through `yfinance`.

---

## 🔧 Data Engineering Pipeline  
All preprocessing is performed using scripts in `/scripts` and `/src`.

### **Pipeline Steps**
1. **Fetch S&P 500 returns (monthly)**  
2. **Geocode HQ → latitude/longitude**  
3. **Map each HQ to a U.S. county**  
4. **Aggregate VIIRS brightness at the county-month level**  
5. **Merge brightness × returns at the firm-month level**  
6. **Compute key variables:**  
   - `avg_rad_month` – brightness level  
   - `brightness_change` – ΔLight this month  
   - `ret_fwd_1m` – next-month return  

The merged file is:  

---

## 📈 Modeling  
We estimate the following regression:

\[
\text{Return}_{i,t+1} = \beta \cdot \Delta Light_{i,t} + \gamma_{t} + \varepsilon_{i,t}
\]

Where:  
- **β** measures how brightness changes predict future returns  
- **γₜ** = *year-month fixed effects* removing market / seasonality  
- Interpretation:  
  - β > 0 → brightening areas perform better  
  - β < 0 → brightness spikes reverse  
  - β ≈ 0 → no predictive link  

---

## 🖥️ Streamlit Dashboard  
The full interactive dashboard is under `app.py` and `pages/`.

### Tabs include:  
- **Overview** – project explanation & key metrics  
- **Ticker Explorer** – firm-level brightness vs. returns  
- **County Explorer** – county-level patterns + R² leaderboards  
- **Regression Analysis** – fixed-effect model results & explanations  
- **Globe View** – interactive 3D map of brightness, returns, and hotspots  

Each tab includes **paragraph explanations** so the audience can understand every visual.

---

## 🧪 Key Findings  
- **Brightness changes alone have weak predictive power** at the county resolution.  
- Fixed-effects R² ≈ **0.20–0.30** is driven mostly by month effects (market-wide).  
- Some individual firms/counties show higher R² (≈15–18%), but not consistent.  
- Likely need **higher-resolution VIIRS radiance grids** (paid) for true signal at HQ-level.

---

## ⚠️ Limitations  
- County-level brightness may be too coarse to capture HQ-specific effects.  
- Satellite noise, clouds, fires, and snow can distort radiance.  
- True local economic signals may require **higher spatial resolution** (500m–750m grids).  
- Next-month stock returns are noisy and driven by many macro factors.

---

## 🧑‍🏫 Acknowledgment  
This project was built for:

**Professor Don Bowen – FIN 377 (Advanced Investments & Data Science)**  
Lehigh University

His guidance, datasets, and modeling framework enabled this research project.

---

## 🏛️ Attribution Summary  
- **VIIRS DNB Data:**  
  Earth Observation Group (EOG) – Payne Institute for Public Policy  
- **Processed VIIRS CSV:**  
  Hosted & indexed by **Jiaxiong Yao**  
- **County shapefiles:** U.S. Census TIGER/Line  
- **Stock returns:** Yahoo Finance via yfinance  
- **Geocoding:** OpenStreetMap Nominatim  

---

## 📂 Repository Structure  

elorevnightlightproject/
│
├── app.py
├── pages/
│ ├── 1_Overview.py
│ ├── 2_Ticker_Explorer.py
│ ├── 3_County_Explorer.py
│ ├── 4_Globe.py
│ └── 5_Regression.py
│
├── data/
│ ├── raw/
│ ├── intermediate/
│ └── final/
│
├── src/
│ ├── preprocess_lights.py
│ ├── preprocess_stocks.py
│ ├── build_panel.py
│ ├── map_firms_to_counties.py
│ └── load_data.py
│
└── scripts/
├── build_all.py
├── fetch_monthly_returns.py
└── add_state_to_sp500.py


---

## 📜 License & Academic Use  
This project is for academic use under the supervision of Professor Bowen.  
Satellite data copyrights belong to their respective owners.

---

## 🚀 Contact & Contributions  
Feel free to reach out with questions or improvements related to the modeling, econometrics, or dashboard.

Created by **Adil Alybaev, Ryan Chrostowski, Kosta Kalavruzos**  
Lehigh University – Finance & Data Science  
