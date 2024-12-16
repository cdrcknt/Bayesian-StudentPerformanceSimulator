# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import plotly.express as px

def create_student_performance_network():
    # Create a simple Bayesian Network for student performance
    model = BayesianNetwork([
        ('difficulty', 'grade'),
        ('intelligence', 'grade'),
        ('intelligence', 'sat'),
        ('grade', 'letter')
    ])
    
    # Define CPDs (Conditional Probability Distributions)
    difficulty_cpd = TabularCPD(
        variable='difficulty',
        variable_card=2,
        values=[[0.6], [0.4]]  # [easy, hard]
    )
    
    intelligence_cpd = TabularCPD(
        variable='intelligence',
        variable_card=2,
        values=[[0.7], [0.3]]  # [low, high]
    )
    
    grade_cpd = TabularCPD(
        variable='grade',
        variable_card=3,
        values=[
            [0.3, 0.05, 0.9, 0.5],
            [0.4, 0.25, 0.08, 0.3],
            [0.3, 0.7, 0.02, 0.2]
        ],
        evidence=['difficulty', 'intelligence'],
        evidence_card=[2, 2]
    )
    
    sat_cpd = TabularCPD(
        variable='sat',
        variable_card=2,
        values=[
            [0.95, 0.2],
            [0.05, 0.8]
        ],
        evidence=['intelligence'],
        evidence_card=[2]
    )
    
    letter_cpd = TabularCPD(
        variable='letter',
        variable_card=3,
        values=[
            [0.9, 0.4, 0.0],
            [0.08, 0.5, 0.2],
            [0.02, 0.1, 0.8]
        ],
        evidence=['grade'],
        evidence_card=[3]
    )
    
    # Add CPDs to model
    model.add_cpds(difficulty_cpd, intelligence_cpd, grade_cpd, sat_cpd, letter_cpd)
    return model

def main():
    st.title("Bayesian Network Random Sampler")
    st.write("This application simulates a student performance model using a Bayesian Network")
    
    # Create sidebar for parameters
    st.sidebar.header("Sampling Parameters")
    n_samples = st.sidebar.slider("Number of samples", 100, 1000, 500)
    
    # Create and sample from the network
    model = create_student_performance_network()
    inference = BayesianModelSampling(model)
    
    if st.button("Generate Samples"):
        # Generate samples
        samples = inference.forward_sample(size=n_samples)
        
        # Display raw data
        st.subheader("Generated Samples")
        st.dataframe(samples.head())
        
        # Create visualizations
        st.subheader("Visualizations")
        
        # Distribution of grades
        fig_grades = px.histogram(samples, x='grade', 
                                title='Distribution of Grades',
                                labels={'grade': 'Grade', 'count': 'Frequency'})
        st.plotly_chart(fig_grades)
        
        # Relationship between intelligence and SAT scores
        fig_intel_sat = px.box(samples, x='intelligence', y='sat',
                             title='Intelligence vs SAT Scores')
        st.plotly_chart(fig_intel_sat)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(samples.describe())

if __name__ == "__main__":
    main()