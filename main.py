"""Main entry point for the Q3A: Quantum-Accelerated AI Agent."""

import streamlit as st

# Simple direct Streamlit app
st.title("QUASAR: Quantum-Accelerated Search and AI Reasoning")
st.write("Welcome to the QUASAR framework demo")

st.header("Quantum Features")
st.write("1. Quantum-accelerated search")
st.write("2. Quantum factorization")
st.write("3. Quantum optimization")

if st.button("Click me"):
    st.success("Button clicked!")