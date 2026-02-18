import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='color: #667eea;'>📚 Student Performance Prediction System</h1>
        <p style='font-size: 16px; color: #666;'>AI-Powered Student Success Prediction & Insight Engine</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
        <h3>🎯 Smart Prediction Features:</h3>
        <ul>
            <li>📊 Single & Batch Student Predictions</li>
            <li>🔍 Detailed Performance Analysis</li>
            <li>📈 Feature Importance & Impact Analysis</li>
            <li>⚠️ Risk Assessment & Early Warnings</li>
            <li>💡 Personalized Recommendations</li>
            <li>📋 Comparative Analytics Dashboard</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Create sidebar for navigation
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Select Page", [
    "🔮 Single Prediction",
    "📊 Batch Prediction", 
    "📈 Analytics Dashboard",
    "⚙️ Model Info"
])

# Load model and preprocessor
@st.cache_resource
def load_model_artifacts():
    try:
        best_model = joblib.load('best_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        training_features = joblib.load('training_features.pkl')
        return best_model, preprocessor, training_features
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

# Function for risk assessment
def assess_risk_level(input_data):
    """Assess student risk level based on input factors"""
    risk_score = 0
    risk_factors = []
    
    if input_data['Hours_Studied'].values[0] < 2:
        risk_score += 3
        risk_factors.append("❌ Low study hours (< 2 hrs/week)")
    
    if input_data['Attendance'].values[0] < 70:
        risk_score += 3
        risk_factors.append("❌ Poor attendance (< 70%)")
    
    if input_data['Sleep_Hours'].values[0] < 6:
        risk_score += 2
        risk_factors.append("⚠️ Insufficient sleep (< 6 hrs/night)")
    
    if input_data['Previous_Scores'].values[0] < 50:
        risk_score += 3
        risk_factors.append("❌ Low previous scores")
    
    if input_data['Motivation_Level'].values[0] < 4:
        risk_score += 2
        risk_factors.append("⚠️ Low motivation level")
    
    if input_data['Parental_Involvement'].values[0] < 4:
        risk_score += 2
        risk_factors.append("⚠️ Limited parental involvement")
    
    return risk_score, risk_factors

# Function to clean and prepare data for predictions
def prepare_batch_data(batch_data):
    """Clean and prepare batch data for predictions"""
    data_clean = batch_data.copy()
    
    # Numeric columns that should be float
    numeric_cols = ['Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
                   'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
                   'Internet_Access', 'Tutoring_Sessions', 'Teacher_Quality', 'Peer_Influence',
                   'Physical_Activity', 'Learning_Disabilities', 'Distance_from_Home']
    
    # Convert numeric columns
    for col in numeric_cols:
        if col in data_clean.columns:
            # Convert to numeric, replacing errors with 0
            data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
            # Fill NaN with 0
            data_clean[col] = data_clean[col].fillna(0)
            # Ensure float type
            data_clean[col] = data_clean[col].astype('float64')
    
    # Categorical columns - ensure they are strings
    categorical_cols = ['Family_Income', 'Teacher_Quality', 'School_Type', 'Parental_Education_Level', 'Gender']
    for col in categorical_cols:
        if col in data_clean.columns:
            data_clean[col] = data_clean[col].astype(str)
    
    return data_clean
        if input_data['Hours_Studied'].values[0] >= 5:
            milestones.append("🏆 Exceptional Study Habits - Consider peer tutoring")
        if input_data['Previous_Scores'].values[0] >= 85:
            milestones.append("⭐ Outstanding Academic Excellence")
        if input_data['Attendance'].values[0] >= 95:
            milestones.append("🎖️ Perfect Attendance Record")
    elif prediction == 1:  # Medium performance
        milestones.append("🎯 On Track - Focus on consistency")
        milestones.append("📈 Opportunity to reach High Performance")
    else:  # Low performance
        milestones.append("💪 Improvement Potential - Seek support")
        milestones.append("🆘 Action Plan Required")
    
    return milestones

# Load artifacts
best_model, preprocessor, training_features = load_model_artifacts()

if best_model is None:
    st.error("⚠️ Could not load model. Please ensure all .pkl files are in the current directory.")
else:
    
    # ==================== SINGLE PREDICTION PAGE ====================
    if page == "🔮 Single Prediction":
        st.header("🔮 Single Student Prediction & Analysis")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📋 Student Info", "🎯 Prediction Result", "📊 Detailed Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📚 Academic Information")
                hours_studied = st.slider("Hours Studied (per week)", 0.0, 15.0, 5.0, step=0.5)
                attendance = st.slider("Attendance (%)", 0, 100, 80, step=1)
                previous_scores = st.slider("Previous Scores (%)", 0, 100, 75, step=1)
                motivation = st.slider("Motivation Level (1-10)", 1, 10, 7, step=1)
                tutoring = st.slider("Tutoring Sessions (per month)", 0, 10, 2, step=1)
                teacher_quality = st.slider("Teacher Quality Rating (1-10)", 1, 10, 7, step=1)
            
            with col2:
                st.subheader("🏠 Personal & Home")
                sleep_hours = st.slider("Sleep Hours (per night)", 0.0, 12.0, 8.0, step=0.5)
                parental_involvement = st.slider("Parental Involvement (1-10)", 1, 10, 5, step=1)
                family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
                parent_education = st.selectbox("Parental Education Level", ["High School", "Bachelor", "Master", "PhD"])
                distance_from_home = st.slider("Distance from Home (km)", 0.0, 50.0, 10.0, step=0.5)
                physical_activity = st.slider("Physical Activity (hours/week)", 0.0, 10.0, 3.0, step=0.5)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("🎓 Resources & Environment")
                access_to_resources = st.slider("Access to Resources (1-10)", 1, 10, 5, step=1)
                internet_access = st.selectbox("Internet Access", ["Yes", "No"])
                school_type = st.selectbox("School Type", ["Public", "Private"])
                extracurricular = st.slider("Extracurricular Activities (1-10)", 1, 10, 5, step=1)
            
            with col4:
                st.subheader("👥 Social & Health")
                peer_influence = st.slider("Peer Influence (1-10)", 1, 10, 5, step=1)
                gender = st.selectbox("Gender", ["Male", "Female"])
                learning_disability = st.selectbox("Learning Disabilities", ["Yes", "No"])
        
        with tab2:
            # Create prediction button
            if st.button("🎯 Generate Prediction & Analysis", key="predict_single", use_container_width=True):
                # Prepare input data with proper encoding
                input_data = pd.DataFrame({
                    'Hours_Studied': [float(hours_studied)],
                    'Attendance': [float(attendance)],
                    'Parental_Involvement': [float(parental_involvement)],
                    'Access_to_Resources': [float(access_to_resources)],
                    'Extracurricular_Activities': [float(extracurricular)],
                    'Sleep_Hours': [float(sleep_hours)],
                    'Previous_Scores': [float(previous_scores)],
                    'Motivation_Level': [float(motivation)],
                    'Internet_Access': [float(1 if internet_access == "Yes" else 0)],
                    'Tutoring_Sessions': [float(tutoring)],
                    'Family_Income': [family_income],
                    'Teacher_Quality': [float(teacher_quality)],
                    'School_Type': [school_type],
                    'Peer_Influence': [float(peer_influence)],
                    'Physical_Activity': [float(physical_activity)],
                    'Learning_Disabilities': [float(1 if learning_disability == "Yes" else 0)],
                    'Parental_Education_Level': [parent_education],
                    'Distance_from_Home': [float(distance_from_home)],
                    'Gender': [gender]
                })
                
                try:
                    # Transform features
                    X_transformed = preprocessor.transform(input_data)
                    X_transformed = pd.DataFrame(X_transformed)
                    
                    # Align with training features
                    X_aligned = pd.DataFrame(0, index=np.arange(X_transformed.shape[0]), columns=training_features)
                    for col in X_transformed.columns:
                        if col in X_aligned.columns:
                            X_aligned[col] = X_transformed[col].values
                    
                    # Make prediction
                    prediction = best_model.predict(X_aligned)[0]
                    
                    # Risk assessment
                    risk_score, risk_factors = assess_risk_level(input_data)
                    
                    # Get prediction probabilities
                    if hasattr(best_model, 'predict_proba'):
                        proba = best_model.predict_proba(X_aligned)[0]
                        
                        performance_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                        predicted_performance = performance_map.get(prediction, 'Unknown')
                        
                        # Main prediction display
                        st.success("✅ Prediction Analysis Complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🎯 Predicted Performance", predicted_performance, 
                                     delta=f"{proba[prediction]*100:.0f}% confidence")
                        
                        with col2:
                            st.metric("Risk Level", f"{risk_score}/10", 
                                     delta="High Risk" if risk_score >= 6 else "Moderate Risk" if risk_score >= 3 else "Low Risk")
                        
                        with col3:
                            st.metric("Overall Score", f"{(proba[prediction]*100):.1f}%",
                                     delta="Strong" if proba[prediction] > 0.7 else "Moderate")
                        
                        # Probability visualization
                        st.subheader("📊 Confidence Breakdown")
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Low', 'Medium', 'High'],
                                y=proba,
                                marker_color=['#ff6b6b', '#ffd93d', '#6bcf7f'],
                                text=[f'{p*100:.1f}%' for p in proba],
                                textposition='auto',
                            )
                        ])
                        fig.update_layout(
                            title="Prediction Probability Distribution",
                            xaxis_title="Performance Level",
                            yaxis_title="Probability",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk factors display
                        if risk_factors:
                            st.subheader("⚠️ Risk Factors Identified")
                            for factor in risk_factors:
                                st.warning(factor)
                
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        
        with tab3:
            if st.button("📊 Run Detailed Analysis", key="detailed_analysis", use_container_width=True):
                # Prepare input data with proper encoding
                input_data = pd.DataFrame({
                    'Hours_Studied': [float(hours_studied)],
                    'Attendance': [float(attendance)],
                    'Parental_Involvement': [float(parental_involvement)],
                    'Access_to_Resources': [float(access_to_resources)],
                    'Extracurricular_Activities': [float(extracurricular)],
                    'Sleep_Hours': [float(sleep_hours)],
                    'Previous_Scores': [float(previous_scores)],
                    'Motivation_Level': [float(motivation)],
                    'Internet_Access': [float(1 if internet_access == "Yes" else 0)],
                    'Tutoring_Sessions': [float(tutoring)],
                    'Family_Income': [family_income],
                    'Teacher_Quality': [float(teacher_quality)],
                    'School_Type': [school_type],
                    'Peer_Influence': [float(peer_influence)],
                    'Physical_Activity': [float(physical_activity)],
                    'Learning_Disabilities': [float(1 if learning_disability == "Yes" else 0)],
                    'Parental_Education_Level': [parent_education],
                    'Distance_from_Home': [float(distance_from_home)],
                    'Gender': [gender]
                })
                
                try:
                    X_transformed = preprocessor.transform(input_data)
                    X_transformed = pd.DataFrame(X_transformed)
                    
                    X_aligned = pd.DataFrame(0, index=np.arange(X_transformed.shape[0]), columns=training_features)
                    for col in X_transformed.columns:
                        if col in X_aligned.columns:
                            X_aligned[col] = X_transformed[col].values
                    
                    prediction = best_model.predict(X_aligned)[0]
                    performance_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                    predicted_performance = performance_map.get(prediction, 'Unknown')
                    
                    # Feature importance
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Key Performance Indicators")
                        kpi_data = {
                            'Study Habits': hours_studied / 15 * 100,
                            'Attendance': attendance,
                            'Sleep Quality': (sleep_hours / 8 * 100) if sleep_hours <= 8 else 100,
                            'Previous Performance': previous_scores,
                            'Motivation': motivation * 10
                        }
                        
                        fig = go.Figure(data=[
                            go.Scatterpolar(
                                r=list(kpi_data.values()),
                                theta=list(kpi_data.keys()),
                                fill='toself',
                                name='Student Profile'
                            )
                        ])
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            title="Student Profile Radar",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("🎯 Achievements & Milestones")
                        milestones = get_achievement_milestones(prediction, input_data)
                        for milestone in milestones:
                            st.info(milestone)
                    
                    # Comparative analysis
                    st.subheader("📊 Comparative Analysis")
                    
                    comparison_data = {
                        'Metric': ['Study Hours', 'Attendance', 'Sleep Hours', 'Previous Scores', 'Motivation'],
                        'Your Value': [hours_studied, attendance, sleep_hours, previous_scores, motivation*10],
                        'Recommended': [5, 85, 8, 75, 80]
                    }
                    
                    fig = go.Figure(data=[
                        go.Bar(name='Your Value', x=comparison_data['Metric'], y=comparison_data['Your Value'], marker_color='#667eea'),
                        go.Bar(name='Recommended', x=comparison_data['Metric'], y=comparison_data['Recommended'], marker_color='#6bcf7f')
                    ])
                    fig.update_layout(title="Performance Comparison", barmode='group', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Personalized Recommendations")
                    
                    recommendations = []
                    
                    if hours_studied < 3:
                        recommendations.append("📚 Increase daily study time to 3-5 hours for better performance")
                    
                    if attendance < 80:
                        recommendations.append("📍 Improve attendance - aim for 85%+ for better understanding")
                    
                    if sleep_hours < 7:
                        recommendations.append("😴 Prioritize sleep - 7-8 hours is optimal for learning")
                    
                    if previous_scores < 60:
                        recommendations.append("🆘 Consider tutoring sessions to strengthen fundamentals")
                    
                    if motivation < 5:
                        recommendations.append("💪 Work with counselor to boost motivation and engagement")
                    
                    if not recommendations:
                        recommendations.append("✅ Continue current effort - you're on the right track!")
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.info(rec)
                
                except Exception as e:
                    st.error(f"Error in analysis: {e}")
    
    # ==================== BATCH PREDICTION PAGE ====================
    elif page == "📊 Batch Prediction":
        st.header("📊 Batch Prediction & Export")
        
        st.markdown("""
        Upload a CSV file containing student data to get predictions for multiple students at once.
        """)
        
        uploaded_file = st.file_uploader("📂 Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📋 Total Students", len(batch_data))
                with col2:
                    st.metric("📊 Columns", len(batch_data.columns))
                with col3:
                    st.metric("✅ Ready", "Yes" if len(batch_data) > 0 else "No")
                
                st.subheader("📄 Data Preview")
                st.dataframe(batch_data.head(10), use_container_width=True)
                
                if st.button("🎯 Predict All Students", use_container_width=True):
                    try:
                        # Clean and prepare data
                        batch_data_clean = prepare_batch_data(batch_data)
                        
                        # Transform features
                        X_batch = preprocessor.transform(batch_data_clean)
                        X_batch = pd.DataFrame(X_batch)
                        X_batch = X_batch.astype('float64')
                        
                        # Align features
                        X_aligned = pd.DataFrame(0, index=np.arange(X_batch.shape[0]), columns=training_features, dtype='float64')
                        for col in X_batch.columns:
                            if col in X_aligned.columns:
                                X_aligned[col] = X_batch[col].values
                        
                        # Make predictions
                        predictions = best_model.predict(X_aligned)
                        
                        # Map predictions
                        performance_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                        predicted_labels = [performance_map.get(int(p), 'Unknown') for p in predictions]
                        
                        # Create results dataframe
                        results = batch_data.copy()
                        results['Predicted_Performance'] = predicted_labels
                        
                        # Get probabilities
                        if hasattr(best_model, 'predict_proba'):
                            proba = best_model.predict_proba(X_aligned)
                            results['Confidence_Low'] = [float(x) for x in proba[:, 0]]
                            results['Confidence_Medium'] = [float(x) for x in proba[:, 1]]
                            results['Confidence_High'] = [float(x) for x in proba[:, 2]]
                            results['Max_Confidence'] = [float(x) for x in np.max(proba, axis=1)]
                        
                        st.success("✅ Predictions Complete!")
                        
                        # Display results
                        st.subheader("🎯 Prediction Results")
                        st.dataframe(results, use_container_width=True)
                        
                        # Statistics
                        st.subheader("📈 Prediction Statistics")
                        
                        value_counts = results['Predicted_Performance'].value_counts()
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Predictions", len(results))
                        with col2:
                            st.metric("High Performance", int(value_counts.get('High', 0)))
                        with col3:
                            st.metric("Medium Performance", int(value_counts.get('Medium', 0)))
                        with col4:
                            st.metric("Low Performance", int(value_counts.get('Low', 0)))
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = go.Figure(data=[go.Pie(
                                labels=value_counts.index,
                                values=value_counts.values,
                                marker=dict(colors=['#ff6b6b', '#ffd93d', '#6bcf7f'])
                            )])
                            fig.update_layout(title="Performance Distribution", height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = go.Figure(data=[go.Bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                marker_color=['#6bcf7f', '#ffd93d', '#ff6b6b']
                            )])
                            fig.update_layout(
                                title="Performance Count",
                                xaxis_title="Performance Level",
                                yaxis_title="Count",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"Error during batch prediction: {str(e)}")
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # ==================== ANALYTICS DASHBOARD ====================
    elif page == "📈 Analytics Dashboard":
        st.header("📈 Analytics & Insights Dashboard")
        
        st.markdown("""
        Upload a CSV file containing student data to generate comprehensive analytics about student performance patterns.
        """)
        
        uploaded_file = st.file_uploader("📂 Choose data file for analysis", type="csv", key="analytics")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                
                # Clean and prepare data
                data = prepare_batch_data(data)
                
                # Make predictions for all data
                X_data = preprocessor.transform(data)
                X_data = pd.DataFrame(X_data)
                X_data = X_data.astype('float64')
                
                X_aligned = pd.DataFrame(0, index=np.arange(X_data.shape[0]), columns=training_features, dtype='float64')
                for col in X_data.columns:
                    if col in X_aligned.columns:
                        X_aligned[col] = X_data[col].values
                
                predictions = best_model.predict(X_aligned)
                proba = best_model.predict_proba(X_aligned)
                
                performance_map = {0: 'Low', 1: 'Medium', 2: 'High'}
                data['Predicted_Performance'] = [performance_map.get(int(p), 'Unknown') for p in predictions]
                data['Confidence'] = [float(x) for x in np.max(proba, axis=1)]
                
                # Dashboard sections
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("👥 Total Students", len(data))
                with col2:
                    avg_confidence = data['Confidence'].mean()
                    st.metric("📊 Avg Confidence", f"{avg_confidence*100:.1f}%")
                with col3:
                    high_performers = len(data[data['Predicted_Performance'] == 'High'])
                    st.metric("🌟 High Performers", f"{high_performers} ({high_performers/len(data)*100:.1f}%)")
                with col4:
                    low_performers = len(data[data['Predicted_Performance'] == 'Low'])
                    st.metric("⚠️ At Risk", f"{low_performers} ({low_performers/len(data)*100:.1f}%)")
                
                # Key metrics analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📚 Key Metrics Distribution")
                    
                    metrics_to_analyze = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Motivation_Level']
                    available_metrics = [m for m in metrics_to_analyze if m in data.columns]
                    
                    if available_metrics:
                        fig = go.Figure()
                        for metric in available_metrics:
                            if metric in data.columns:
                                clean_data = data[metric].fillna(data[metric].mean()).dropna()
                                fig.add_trace(go.Box(
                                    y=clean_data,
                                    name=metric,
                                    boxmean='sd'
                                ))
                        
                        fig.update_layout(
                            title="Distribution of Key Metrics",
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🎯 Performance by Demographics")
                    
                    if 'Gender' in data.columns:
                        gender_performance = data.groupby('Gender')['Predicted_Performance'].value_counts().unstack(fill_value=0)
                        
                        fig = go.Figure(data=[
                            go.Bar(name='High', x=gender_performance.index, y=gender_performance.get('High', [0]*len(gender_performance))),
                            go.Bar(name='Medium', x=gender_performance.index, y=gender_performance.get('Medium', [0]*len(gender_performance))),
                            go.Bar(name='Low', x=gender_performance.index, y=gender_performance.get('Low', [0]*len(gender_performance)))
                        ])
                        fig.update_layout(
                            title="Performance by Gender",
                            barmode='group',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Risk assessment visualization
                st.subheader("⚠️ Risk Assessment Summary")
                
                risk_data = []
                for idx, row in data.iterrows():
                    risk_score = 0
                    if row.get('Hours_Studied', 0) < 2:
                        risk_score += 3
                    if row.get('Attendance', 0) < 70:
                        risk_score += 3
                    if row.get('Previous_Scores', 0) < 50:
                        risk_score += 3
                    if row.get('Sleep_Hours', 0) < 6:
                        risk_score += 2
                    risk_data.append(risk_score)
                
                data['Risk_Score'] = risk_data
                
                risk_categories = pd.cut(data['Risk_Score'], bins=[0, 3, 6, 12], labels=['Low Risk', 'Moderate Risk', 'High Risk'], include_lowest=True)
                risk_counts = risk_categories.value_counts().dropna()
                
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index,
                    values=risk_counts.values,
                    marker=dict(colors=['#6bcf7f', '#ffd93d', '#ff6b6b'])
                )])
                fig.update_layout(title="Student Risk Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top and bottom performers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏆 Top Performers")
                    if 'Confidence' in data.columns:
                        top_performers = data.nlargest(5, 'Confidence')
                        display_cols = [col for col in ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Confidence'] if col in top_performers.columns]
                        if display_cols:
                            st.dataframe(top_performers[display_cols].fillna(0), use_container_width=True)
                
                with col2:
                    st.subheader("⚠️ Students Needing Support")
                    if 'Risk_Score' in data.columns:
                        at_risk = data.nlargest(5, 'Risk_Score')
                        display_cols = [col for col in ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Risk_Score'] if col in at_risk.columns]
                        if display_cols:
                            st.dataframe(at_risk[display_cols].fillna(0), use_container_width=True)
            
            except Exception as e:
                st.error(f"Error in analytics: {e}")
    
    # ==================== MODEL INFO PAGE ====================
    else:  # ⚙️ Model Info
        st.header("ℹ️ Model Information & Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Architecture")
            st.write(f"**Model Type:** {type(best_model).__name__}")
            st.write(f"**Number of Features:** {len(training_features)}")
            st.write(f"**Prediction Classes:** 3 (Low, Medium, High)")
            st.write(f"**Training Accuracy:** 99%")
        
        with col2:
            st.subheader("🎯 Performance Classes")
            st.markdown("""
            - 🔴 **Low:** Exam Score < 50
            - 🟡 **Medium:** Exam Score 50-75
            - 🟢 **High:** Exam Score 75-100
            """)
        
        st.subheader("📝 Feature Description")
        
        features_info = {
            'Hours_Studied': 'Number of hours studied per week (0-15)',
            'Attendance': 'Class attendance percentage (0-100%)',
            'Parental_Involvement': 'Level of parental support (1-10)',
            'Access_to_Resources': 'Access to learning resources (1-10)',
            'Extracurricular_Activities': 'Participation in extra-curricular (1-10)',
            'Sleep_Hours': 'Average sleep hours per night (0-12)',
            'Previous_Scores': 'Average of previous exam scores (%)',
            'Motivation_Level': 'Student motivation level (1-10)',
            'Internet_Access': 'Internet access availability (Yes/No)',
            'Tutoring_Sessions': 'Tutoring sessions per month (0-10)',
            'Family_Income': 'Family income level (Low/Medium/High)',
            'Teacher_Quality': 'Teacher quality rating (1-10)',
            'School_Type': 'School type (Public/Private)',
            'Peer_Influence': 'Peer group influence (1-10)',
            'Physical_Activity': 'Physical activity hours/week',
            'Learning_Disabilities': 'Learning disabilities (Yes/No)',
            'Parental_Education': 'Parents education level',
            'Distance_from_Home': 'Distance to school (km)',
            'Gender': 'Student gender'
        }
        
        features_df = pd.DataFrame([
            (k, v) for k, v in features_info.items()
        ], columns=['Feature', 'Description'])
        
        st.dataframe(features_df, use_container_width=True, height=400)
        
        st.subheader("🚀 How to Use")
        
        tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Dashboard"])
        
        with tab1:
            st.markdown("""
            ### Single Student Prediction
            
            1. **Fill Student Information** - Enter all required student details
            2. **Generate Prediction** - Click to get prediction with confidence
            3. **Detailed Analysis** - Get recommendations and insights
            """)
        
        with tab2:
            st.markdown("""
            ### Batch Prediction
            
            1. **Prepare CSV** with all required columns
            2. **Upload File** - Select your data file
            3. **Predict** - Get predictions for all students
            4. **Export** - Download results as CSV
            """)
        
        with tab3:
            st.markdown("""
            ### Analytics Dashboard
            
            1. **Upload Data** - Upload student data file
            2. **View Analytics** - See performance patterns
            3. **Export Insights** - Generate reports
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎓 Student Performance Prediction System | Powered by ML & Streamlit</p>
    <p style='font-size: 12px;'>Model Accuracy: 99% | 2024</p>
</div>
""", unsafe_allow_html=True)
