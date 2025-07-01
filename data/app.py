import streamlit as st
climate_data.csv")

# --- Preprocess ---
df['avg_temp'] = (df['Max_Temperature_C'] + df['Min_Temperature_C']) / 2
df['year'] = pd.to_datetime(df['Year'], errors='coerce').dt.year
df = df.dropna(subset=['year', 'Total_Rainfall_mm'])

# --- Features and Target ---
X = df[['year', 'Total_Rainfall_mm']]
y = df['avg_temp']

# --- Train model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Streamlit UI ---
st.title("🌍 Tanzania Climate Prediction App")
st.write("Explore historical climate data and predict future average temperatures.")

# --- EDA: Line Plot ---
st.subheader("📈 Historical Temperature Trends")
fig, ax = plt.subplots()
df.groupby("year")['avg_temp'].mean().plot(ax=ax)
ax.set_ylabel("Average Temperature (°C)")
st.pyplot(fig)

# --- Prediction Form ---
st.subheader("🔮 Predict Future Avg Temperature")

year = st.slider("Select a Year", min_value=2025, max_value=2100, value=2030)
rainfall = st.number_input("Enter Estimated Rainfall (mm)", value=float(df['Total_Rainfall_mm'].mean()))

# --- Make Prediction ---
input_data = pd.DataFrame({'year': [year], 'Total_Rainfall_mm': [rainfall]})
prediction = model.predict(input_data)[0]

st.success(f"Predicted Avg Temperature for {year} is **{prediction:.2f}°C**")

# --- Data Preview ---
st.subheader("🧾 Dataset Preview")
st.dataframe(df.head())