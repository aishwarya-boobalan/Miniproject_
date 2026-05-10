import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Load model
model = pickle.load(open("house_model.pkl", "rb"))

# Load data
data = pd.read_csv("housing.csv")

# Page config
st.set_page_config(
    page_title="House Price Dashboard",
    page_icon="🏠",
    layout="wide"
)

# Title
st.markdown(
    "<h1 style='text-align:center;color:#4CAF50;'>🏠 AI House Price Prediction Dashboard</h1>",
    unsafe_allow_html=True
)

st.write("Predict house prices using Machine Learning.")

# Sidebar
st.sidebar.header("Enter House Details")

area = st.sidebar.slider("Area", 500, 10000, 2000)

bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)

bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)

stories = st.sidebar.slider("Stories", 1, 5, 2)

parking = st.sidebar.slider("Parking", 0, 5, 1)

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Houses", len(data))

col2.metric("Average Price", f"₹ {int(data['price'].mean()):,}")

col3.metric("Maximum Price", f"₹ {int(data['price'].max()):,}")

st.markdown("---")

# Prediction result
st.subheader("Predicted Price")

if st.sidebar.button("Predict Price"):

    features = np.array([[area, bedrooms, bathrooms, stories, parking]])

    prediction = model.predict(features)[0]

    st.success(f"Estimated Price: ₹ {prediction:,.2f}")
else:

    st.info("Click 'Predict Price' to get an estimate")

# Charts
st.markdown("---")

col4, col5 = st.columns(2)

# PIE CHART
with col4:

    bedroom_counts = data['bedrooms'].value_counts()

    fig1 = px.pie(
        values=bedroom_counts.values,
        names=bedroom_counts.index,
        title="Bedroom Distribution"
    )

    st.plotly_chart(fig1, use_container_width=True)

# BAR CHART
with col5:

    avg_price = data.groupby("stories")["price"].mean().reset_index()

    fig2 = px.bar(
        avg_price,
        x="stories",
        y="price",
        title="Average Price by Stories"
    )

    st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")

st.caption("Developed by Aishwarya")