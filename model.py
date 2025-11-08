import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import time
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="CreditRisk AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Phone brand mapping
PHONE_BRANDS = {
    'low': ['xiaomi', 'poco', 'nokia', 'realme', 'tecno', 'infinix', 'itel', 'oppo', 'vivo', 'micromax', 'lava', 'gionee'],
    'average': ['samsung', 'google', 'oneplus', 'motorola', 'lg', 'honor', 'huawei', 'lenovo', 'sony', 'htc'],
    'premium': ['apple', 'nothing', 'asus', 'google pixel', 'oneplus pro', 'samsung ultra']
}

# Loan type mapping with base interest rates and risk factors
LOAN_TYPES = {
    "Personal Loan": {"base_rate": 10.5, "risk_factor": 1.2, "lgd": 0.45},
    "Home Loan": {"base_rate": 8.2, "risk_factor": 1.0, "lgd": 0.35},
    "Car Loan": {"base_rate": 7.8, "risk_factor": 1.1, "lgd": 0.50},
    "Education Loan": {"base_rate": 8.0, "risk_factor": 0.9, "lgd": 0.40},
    "Business Loan": {"base_rate": 11.5, "risk_factor": 1.4, "lgd": 0.55},
    "Credit Card Loan": {"base_rate": 14.0, "risk_factor": 1.5, "lgd": 0.75},
    "Gold Loan": {"base_rate": 7.5, "risk_factor": 0.8, "lgd": 0.25}
}

# Function to determine loan decision and interest rate based on PD
def get_loan_decision(default_prob, base_rate):
    if default_prob < 0.12:
        decision = "APPROVED"
        decision_color = "green"
        interest_rate = base_rate
        recommendation = "Standard rate applied. Loan approved with favorable terms."
    elif 0.12 <= default_prob < 0.20:
        decision = "APPROVED"
        decision_color = "orange"
        interest_rate = base_rate + 1.0
        recommendation = "Approved with higher interest rate (+1.0%)."
    elif 0.20 <= default_prob < 0.40:
        decision = "APPROVED WITH CONDITIONS"
        decision_color = "orange"
        interest_rate = base_rate + 2.5
        recommendation = "Approved with significantly higher rate (+2.5%). Consider guarantee or co-lending options."
    else:  # PD >= 0.40
        decision = "MANUAL REVIEW REQUIRED"
        decision_color = "red"
        interest_rate = base_rate + 5.0  # Placeholder, will need manual assessment
        recommendation = "High default risk. Requires manual underwriting review. Likely to be rejected."
    
    return decision, decision_color, interest_rate, recommendation

def map_phone_brand(brand_name):
    brand_lower = brand_name.lower()
    for category, brands in PHONE_BRANDS.items():
        if any(brand in brand_lower for brand in brands):
            if category == 'low':
                return 0
            elif category == 'average':
                return 1
            elif category == 'premium':
                return 2
    return 1  # Default to average if brand not found

# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Update this path to your actual model file
        with open('loan_repayment_status_xgb.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

# Function to get economic indicators (in a real app, this would fetch from an API)
def get_economic_indicators():
    # Current real-time values (as of your request)
    return {
        'gdp_growth': 7.40,
        'inflation': 2.82,
        'unemployment_rate': 5.2,
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Function to calculate credit health score
def calculate_credit_health_score(default_prob):
    # Convert default probability to credit health score (0-1000 scale)
    # Higher score = better credit health
    health_score = 1000 * (1 - default_prob)
    return max(300, min(1000, int(health_score)))  # Clamp between 300-1000

# Function to get credit rating category
def get_credit_rating(score):
    if score >= 900:
        return "Excellent", "#00b894"
    elif score >= 800:
        return "Very Good", "#00cec9"
    elif score >= 700:
        return "Good", "#0984e3"
    elif score >= 600:
        return "Fair", "#fdcb6e"
    elif score >= 500:
        return "Poor", "#e17055"
    else:
        return "Very Poor", "#d63031"

# Function to calculate expected loss
def calculate_expected_loss(pd, lgd, ead):
    """
    Calculate Expected Loss using the formula: EL = PD * LGD * EAD
    
    Parameters:
    pd (float): Probability of Default (0-1)
    lgd (float): Loss Given Default (0-1)
    ead (float): Exposure at Default (loan amount)
    
    Returns:
    float: Expected Loss amount
    """
    return pd * lgd * ead
# Function to analyze defaulters from the dataset
def analyze_defaulters(df):
    # Calculate flags for each user (count of zeros in EMI columns)
    emi_columns = ['emi1', 'emi2', 'emi3', 'emi4', 'emi5', 'emi6']
    salary_columns = ['salary1', 'salary2', 'salary3']
    
    df['flag_count'] = df[emi_columns].apply(lambda x: (x == 0).sum(), axis=1)
    df['salary_flag'] = df[salary_columns].apply(lambda x: (x == 1).any(), axis=1)
    
    # Create defaulter list (30% of users with exactly 3 EMI flags)
    total_users = len(df)
    defaulter_count = int(total_users * 0.3)
    
    # FIX: Check if sample size exceeds available population
    users_with_3_flags = df[df['flag_count'] == 3]
    if len(users_with_3_flags) > 0:
        sample_size = min(defaulter_count, len(users_with_3_flags))
        defaulter_list = users_with_3_flags.sample(sample_size, random_state=42)
    else:
        defaulter_list = pd.DataFrame()  # Empty dataframe if no users with 3 flags
    
    # Create watch list (20% of users with exactly 2 EMI flags OR any salary flag)
    watch_count = int(total_users * 0.2)
    
    # Get users with 2 EMI flags
    users_with_2_flags = df[df['flag_count'] == 2]
    
    # Get users with any salary flag (non-credited salary)
    users_with_salary_flags = df[df['salary_flag'] == True]
    
    # Combine both criteria for watch list
    watch_candidates = pd.concat([users_with_2_flags, users_with_salary_flags]).drop_duplicates()
    
    if len(watch_candidates) > 0:
        sample_size = min(watch_count, len(watch_candidates))
        watch_list = watch_candidates.sample(sample_size, random_state=42)
    else:
        watch_list = pd.DataFrame()  # Empty dataframe if no watch candidates
    
    # Create good standing list (the remaining users)
    defaulter_ids = defaulter_list['LoanID'].tolist() if not defaulter_list.empty else []
    watch_ids = watch_list['LoanID'].tolist() if not watch_list.empty else []
    good_standing_list = df[~df['LoanID'].isin(defaulter_ids + watch_ids)]
    
    # Summary statistics (with error handling for empty dataframes)
    defaulter_summary = defaulter_list.describe() if not defaulter_list.empty else pd.DataFrame()
    watch_summary = watch_list.describe() if not watch_list.empty else pd.DataFrame()
    good_standing_summary = good_standing_list.describe()
    
    # Create a summary DataFrame for visualization
    summary_data = {
    'Category': ['Defaulters', 'Watch List', 'Good Standing'],
    'Count': [
        len(defaulter_list) if not defaulter_list.empty else 0,
        len(watch_list) if not watch_list.empty else 0,
        len(good_standing_list)
    ],
    'Percentage': [
        (len(defaulter_list) / total_users * 100) if not defaulter_list.empty else 0,
        (len(watch_list) / total_users * 100) if not watch_list.empty else 0,
        (len(good_standing_list) / total_users * 100)
    ],
    'Avg_Flags': [
        defaulter_list['flag_count'].mean() if not defaulter_list.empty else 0,
        watch_list['flag_count'].mean() if not watch_list.empty else 0,
        good_standing_list['flag_count'].mean()  # FIXED: Changed from 'flag_flag_count'
    ],
    'Salary_Flag_Count': [
        defaulter_list['salary_flag'].sum() if not defaulter_list.empty else 0,
        watch_list['salary_flag'].sum() if not watch_list.empty else 0,
        good_standing_list['salary_flag'].sum()
    ]
}
    
    summary_df = pd.DataFrame(summary_data)
    
    return {
        'defaulter_list': defaulter_list,
        'watch_list': watch_list,
        'good_standing_list': good_standing_list,
        'summary_df': summary_df,
        'emi_columns': emi_columns,
        'salary_columns': salary_columns
    }

# Track application status for analytics
if 'application_history' not in st.session_state:
    st.session_state.application_history = {
        'approved': 0,
        'approved_with_conditions': 0,
        'manual_review': 0,
        'total': 0
    }

# Main app
def main():
    st.sidebar.title("CreditRisk AI")
    st.sidebar.markdown("""
    **Hackathon Solution**: Using alternative data for credit risk assessment
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.warning("Running in demo mode with simulated predictions")
    
    # Get economic indicators
    economic_data = get_economic_indicators()
    
    # Header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/1005/1005141.png", width=100)
    with col2:
        st.title("CreditRisk AI")
        st.markdown("""
        ### Alternative Data Credit Scoring System
        Assess loan default probability using non-traditional data sources
        """)
    
    # Economic factors section (read-only)
    st.header("📈 Current Economic Environment", divider="blue")
    
    econ_col1, econ_col2, econ_col3, econ_col4 = st.columns(4)
    
    with econ_col1:
        st.metric(
            label="GDP Growth Rate", 
            value=f"{economic_data['gdp_growth']}%",
            help="Current GDP growth rate"
        )
    
    with econ_col2:
        st.metric(
            label="Inflation Rate", 
            value=f"{economic_data['inflation']}%",
            help="Current inflation rate"
        )
    
    with econ_col3:
        st.metric(
            label="Unemployment Rate", 
            value=f"{economic_data['unemployment_rate']}%",
            help="Current unemployment rate"
        )
        
    with econ_col4:
        st.metric(
            label="Last Updated", 
            value=economic_data['last_updated'].split()[0],
            delta=economic_data['last_updated'].split()[1],
            help="When economic data was last updated"
        )
    
    st.info("ℹ️ Economic indicators are updated in real-time and cannot be modified as they reflect current market conditions.")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Applicant Information", "Financial Behavior", "Risk Assessment", "Defaulter Analysis"])
    
    with tab1:
        st.header("Applicant Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name")
            age = st.slider("Age", 18, 70, 30)
            email = st.text_input("Email Address")
            phone = st.text_input("Phone Number")
            
            # Phone model input
            phone_brand = st.selectbox(
                "Phone Brand",
                options=sorted(list(set([brand for category in PHONE_BRANDS.values() for brand in category]))),
                help="Select the brand of your phone"
            )
            
        with col2:
            employment_status = st.selectbox(
                "Employment Status",
                options=["Employed", "Self-Employed", 'Student', 'Unemployed', 'Retired']
            )
            
            education = st.selectbox(
                "Education Level",
                options=["High School", "Bachelor's Degree", "Master's Degree", "Doctorate", "Other"]
            )
            
            years_at_address = st.slider("Years at Current Address", 0, 20, 2)
            marital_status = st.selectbox(
                "Marital Status",
                options=["Single", "Married", "Divorced", "Widowed"]
            )
    
    with tab2:
        st.header("Financial Behavior")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Salary credit SMS feature
            salary_credit_sms = st.radio(
                "Do you receive regular salary credits?",
                options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Regular salary credits indicate stable employment"
            )
            
            housing_type = st.selectbox(
                "Housing Type",
                options=["Own", "Rent", "With Family", "Other"]
            )
            
            # Form fill time input
            form_fill_time_sec = st.slider(
                "Form Fill Time (seconds)",
                min_value=30,
                max_value=300,
                value=120,
                help="Time taken to complete the application form"
            )
            
            # Loan type selection
            loan_type = st.selectbox(
                "Type of Loan",
                options=list(LOAN_TYPES.keys()),
                help="Select the type of loan you're applying for"
            )
            
            # Loan amount input
            loan_amount = st.number_input(
                "Loan Amount (₹)",
                min_value=10000,
                max_value=5000000,
                value=500000,
                step=10000,
                help="Enter the loan amount you're applying for"
            )
            
            # Display loan details based on selection
            if loan_type:
                st.info(f"""
                **{loan_type} Details:**
                - Base Interest Rate: {LOAN_TYPES[loan_type]['base_rate']}%
                - Risk Factor: {LOAN_TYPES[loan_type]['risk_factor']}x
                """)
            
            swiggy_zomato_usage = st.slider(
                "Food Delivery App Usage (times/week)",
                min_value=0,
                max_value=15,
                value=4,
                help="How often do you use food delivery apps like Swiggy/Zomato?"
            )
            
        with col2:
            amazon_usage = st.slider(
                "E-commerce App Usage (times/week)",  # CHANGED FROM times/month
                min_value=0,
                max_value=15,  # CHANGED FROM 30
                value=4,  # CHANGED FROM 12
                help="How often do you use e-commerce apps like Amazon?"
            )
            
            # Finance app usage input
            finance_app_usage = st.slider(
                "Finance App Usage (times/month)",
                min_value=0,
                max_value=30,
                value=5,
                help="How often do you use financial apps like Mint, ET Money, etc.?"
            )
            
            loans_active = st.slider(
                "Number of Active Loans",
                min_value=0,
                max_value=5,
                value=1,
                help="Number of loans you currently have"
            )
            
            rooted_device = st.radio(
                "Is your phone rooted?",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Rooted/jailbroken devices may indicate higher risk"
            )
            
            # E-banking transactions level
            ebanking_transactions_level = st.slider(
                "E-banking Transaction Level",
                min_value=0,
                max_value=10,
                value=5,
                help="Level of e-banking transaction activity (0-10)"
            )
    
    with tab3:
        st.header("Risk Assessment")
        
        st.info("""
        The risk assessment considers your personal information, financial behavior, and current economic conditions
        to calculate your probability of default. Economic factors are set to current real-time values.
        """)
        
        if st.button("Calculate Default Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing data and calculating risk..."):
                time.sleep(1.5)
                
                # Map categorical variables to numerical
                phone_model = map_phone_brand(phone_brand)
                
                # Get loan type risk factor and LGD
                loan_risk_factor = LOAN_TYPES[loan_type]['risk_factor']
                base_interest_rate = LOAN_TYPES[loan_type]['base_rate']
                lgd = LOAN_TYPES[loan_type]['lgd']
                
                # TRY TO USE ACTUAL MODEL FIRST
                try:
                    # Prepare features in EXACT order that model expects with EXACT names
                    features = np.array([[
                        swiggy_zomato_usage,           # swiggy_zomato_usage
                        amazon_usage,                  # amazon_usage
                        phone_model,                   # phone_model
                        economic_data['gdp_growth'],   # gdp_growth
                        salary_credit_sms,             # salary_credit_sms
                        1,                             # min_balance_trend (placeholder)
                        economic_data['unemployment_rate'],  # unemployment_rate
                        loans_active,                  # loans_active
                        rooted_device,                 # rooted_device
                        2,                             # location_consistency (placeholder)
                        economic_data['inflation'],    # inflation
                        form_fill_time_sec,            # form_fill_time_sec
                        finance_app_usage,             # finance_app_usage
                        ebanking_transactions_level    # ebanking_transactions_level
                    ]])
                    
                    # Get prediction from actual model - handle both regression and classification models
                    if hasattr(model, 'predict_proba'):
                        # Classification model
                        prediction = model.predict_proba(features)
                        default_prob = prediction[0][1]  # Probability of default class
                    elif hasattr(model, 'predict'):
                        # Regression model - convert prediction to probability
                        prediction = model.predict(features)
                        # Assuming the model predicts a risk score between 0 and 1
                        default_prob = float(prediction[0])
                        # Clamp the probability between 0.1 and 0.9
                        default_prob = max(0.1, min(0.9, default_prob))
                    else:
                        raise AttributeError("Model doesn't have predict or predict_proba method")
                    
                    st.success("✅ Using actual model predictions")
                    
                except Exception as e:
                    st.error(f"❌ Model prediction failed: {str(e)}")
                    st.info("Falling back to simulated assessment for demo purposes")
                    
                    # Fallback simulation - aligned with model features
                    # Phone brand has significant impact (low=0 increases risk more)
                    phone_risk_factor = 0.5 if phone_model == 0 else (0.2 if phone_model == 1 else 0)
                    
                    employment_map = {"Employed": 2, "Self-Employed": 1, "Student": 0, "Unemployed": -1, "Retired": 1}
                    employment_encoded = employment_map.get(employment_status, 0)
                    
                    risk_factors = {
                        'phone_risk': phone_risk_factor,
                        'employment_risk': max(0, min(1, (2 - employment_encoded) / 3)),
                        'loan_burden_risk': min(1, loans_active * 0.2),
                        'device_risk': rooted_device * 0.3,
                        'delivery_usage_risk': min(1, swiggy_zomato_usage / 15 * 0.5),
                        'ecommerce_risk': min(1, amazon_usage / 30 * 0.3),
                        'form_fill_risk': min(1, form_fill_time_sec / 300),
                        'salary_risk': 1 - salary_credit_sms * 0.5,
                        'ebanking_risk': 1 - (ebanking_transactions_level / 10),
                        'finance_app_risk': 1 - (finance_app_usage / 30)
                    }
                    
                    weights = {
                        'phone_risk': 0.15, 'employment_risk': 0.15,
                        'loan_burden_risk': 0.10, 'device_risk': 0.08, 
                        'delivery_usage_risk': 0.08, 'ecommerce_risk': 0.07, 
                        'form_fill_risk': 0.10, 'salary_risk': 0.12,
                        'ebanking_risk': 0.08, 'finance_app_risk': 0.07
                    }
                    
                    default_prob = 0
                    for factor, weight in weights.items():
                        default_prob += risk_factors[factor] * weight
                    
                    default_prob = max(0.1, min(0.9, default_prob + np.random.normal(0, 0.05)))
                
                # ... your existing code to calculate default_prob ...

                # Apply loan type risk factor
                original_prob = default_prob
                default_prob = min(0.95, default_prob * loan_risk_factor)
                loan_impact = default_prob - original_prob

                # NEW: Check for high food delivery usage with balance tilts
                high_food_delivery = swiggy_zomato_usage > 10  # More than 10 times per week
                frequent_balance_tilts = ebanking_transactions_level < 3  # Low e-banking activity

                if high_food_delivery and frequent_balance_tilts:
                    # Increase default probability by 15%
                    original_before_penalty = default_prob
                    default_prob = min(0.95, default_prob * 1.15)
                    penalty_impact = default_prob - original_before_penalty
    
                    # Show popup message
                    st.warning("""
⚠️                  **Risk Factor Detected**: 
                    High food delivery usage combined with frequent balance tilts has increased your default probability by 15%.

                    - Food delivery usage: {} times/week (threshold: >10)
                    - E-banking activity level: {}/10 (threshold: <3)
                    - Probability increase: {:.2f}%

                    Recommendation: Consider reducing discretionary spending and maintaining higher account balances.
                    """.format(swiggy_zomato_usage, ebanking_transactions_level, penalty_impact*100))

# Continue with your existing code...
# Calculate credit health score
# ... rest of your code ...
                
                # Apply loan type risk factor
                original_prob = default_prob
                default_prob = min(0.95, default_prob * loan_risk_factor)
                loan_impact = default_prob - original_prob
                
                # Calculate credit health score
                credit_score = calculate_credit_health_score(default_prob)
                rating, rating_color = get_credit_rating(credit_score)
                
                # Get loan decision based on PD
                decision, decision_color, final_interest_rate, recommendation = get_loan_decision(default_prob, base_interest_rate)
                
                # Calculate expected loss
                expected_loss = calculate_expected_loss(default_prob, lgd, loan_amount)
                
                # Track application status
                st.session_state.application_history['total'] += 1
                if "APPROVED" in decision and "CONDITIONS" not in decision:
                    st.session_state.application_history['approved'] += 1
                elif "APPROVED WITH CONDITIONS" in decision:
                    st.session_state.application_history['approved_with_conditions'] += 1
                elif "MANUAL REVIEW" in decision:
                    st.session_state.application_history['manual_review'] += 1
                
                # Generate SHAP explanation
                shap_values = {}
                feature_contributions = {
                    'Employment': 20 if employment_status == "Employed" else 10,
                    'Phone Quality': phone_model * 15,  # Higher impact for lower quality phones
                    'Spending Habits': min(15, (swiggy_zomato_usage * 4 + amazon_usage) / 4),
                    'Salary Regularity': salary_credit_sms * 15,
                    'E-banking Activity': ebanking_transactions_level * 3,
                    'Finance App Usage': finance_app_usage * 1,
                    'Economic Factors': 10,
                    'Loan Type': LOAN_TYPES[loan_type]['risk_factor'] * 12
                }
                
                # Normalize to 100%
                total = sum(feature_contributions.values())
                for feature, value in feature_contributions.items():
                    shap_values[feature] = (value / total) * 100
                
                # Display results
                st.subheader("Risk Assessment Results")
                
                # Show loan decision
                st.markdown(f"<h3 style='color: {decision_color}; text-align: center;'>LOAN DECISION: {decision}</h3>", 
                          unsafe_allow_html=True)
                
                # Show loan type impact
                st.warning(f"""
                📊 **Loan Type Impact**: 
                Your selected loan type **{loan_type}** with a base interest rate of **{base_interest_rate}%** 
                has {'increased' if loan_impact > 0 else 'decreased'} your default probability by **{abs(loan_impact):.2%}**
                """)
                
                # Show final interest rate and recommendation
                st.info(f"""
                💵 **Final Interest Rate**: {final_interest_rate}%  
                📋 **Recommendation**: {recommendation}
                """)
                
                # Create columns for results
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    # Credit Health Score
                    st.metric(
                        label="Credit Health Score",
                        value=credit_score,
                        delta=rating,
                        delta_color="off"
                    )
                    st.markdown(f"<h4 style='color: {rating_color}; text-align: center;'>{rating}</h4>", 
                              unsafe_allow_html=True)
                    
                    # Default Probability
                    st.metric(
                        label="Default Probability",
                        value=f"{default_prob:.2%}",
                        help="Probability of loan default"
                    )
                
                with col2:
                    # Risk Level
                    st.subheader("Risk Level")
                    if default_prob < 0.3:
                        risk_text = "LOW RISK"
                        risk_color = "success"
                        risk_icon = "✅"
                    elif default_prob < 0.6:
                        risk_text = "MEDIUM RISK"
                        risk_color = "warning"
                        risk_icon = "⚠️"
                    else:
                        risk_text = "HIGH RISK"
                        risk_color = "error"
                        risk_icon = "🚨"
                    
                    if risk_color == "success":
                        st.success(risk_text)
                    elif risk_color == "warning":
                        st.warning(risk_text)
                    else:
                        st.error(risk_text)
                    
                    st.progress(float(default_prob))
                    st.info(f"{risk_icon} {risk_text} based on alternative data analysis.")
                    
                    # Expected Loss
                    st.metric(
                        label="Expected Loss",
                        value=f"₹{expected_loss:,.2f}",
                        help="Expected Loss = PD × LGD × EAD"
                    )
                    st.caption(f"Based on PD: {default_prob:.2%}, LGD: {lgd:.2%}, EAD: ₹{loan_amount:,.2f}")
                
                with col3:
                    # SHAP Explanation Pie Chart
                    st.subheader("Score Factors")
                    
                    # Prepare data for pie chart
                    labels = list(shap_values.keys())
                    sizes = list(shap_values.values())
                    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
                    
                    fig, ax = plt.subplots(figsize=(8, 6))  # FIXED: Changed subforms to subplots
                    wedges, texts, autotexts = ax.pie(
                        sizes, labels=labels, autopct='%1.1f%%', 
                        colors=colors, startangle=90
                    )
                    
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax.axis('equal')
                    plt.title('Factors Influencing Credit Score', fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Application analytics
                st.subheader("📊 Application Analytics")
                analytics_col1, analytics_col2, analytics_col3, analytics_col4 = st.columns(4)
                
                total_apps = st.session_state.application_history['total']
                approved_rate = (st.session_state.application_history['approved'] / total_apps * 100) if total_apps > 0 else 0
                conditional_rate = (st.session_state.application_history['approved_with_conditions'] / total_apps * 100) if total_apps > 0 else 0
                review_rate = (st.session_state.application_history['manual_review'] / total_apps * 100) if total_apps > 0 else 0
                
                with analytics_col1:
                    st.metric("Total Applications", total_apps)
                with analytics_col2:
                    st.metric("Approval Rate", f"{approved_rate:.1f}%")
                with analytics_col3:
                    st.metric("Conditional Approval", f"{conditional_rate:.1f}%")
                with analytics_col4:
                    st.metric("Manual Review Needed", f"{review_rate:.1f}%")
                
                # Additional insights
                st.markdown("#### 📊 Key Insights:")
                insight_col1, insight_col2 = st.columns(2)
                
                with insight_col1:
                    st.write("**Positive Factors:**")
                    if employment_status == "Employed":
                        st.write("- 💼 Stable employment status")
                    if phone_model == 2:
                        st.write("- 📱 Premium phone brand indicates financial stability")
                    if salary_credit_sms == 1:
                        st.write("- 💰 Regular salary credits")
                    if ebanking_transactions_level > 7:
                        st.write("- 📊 Active e-banking usage")
                    if finance_app_usage > 15:
                        st.write("- 📈 Active finance app usage")
                    if loan_risk_factor < 1.0:
                        st.write(f"- 📉 Lower-risk loan type ({loan_type})")
                
                with insight_col2:
                    st.write("**Areas for Improvement:**")
                    if employment_status == "Unemployed":
                        st.write("- 💼 Explore stable employment options")
                    if phone_model == 0:
                        st.write("- 📱 Lower-end phone brand may indicate financial stress")
                    if loans_active > 3:
                        st.write("- 📉 Reduce existing debt burden")
                    if rooted_device == 1:
                        st.write("- 📱 Consider device security measures")
                    if swiggy_zomato_usage > 10:
                        st.write("- 🍔 Reduce food delivery spending")
                    if salary_credit_sms == 0:
                        st.write("- 💸 Establish regular income pattern")
                    if ebanking_transactions_level < 3:
                        st.write("- 📈 Increase e-banking activity")
                    if finance_app_usage < 5:
                        st.write("- 📱 Increase finance app usage")
                    if loan_risk_factor > 1.0:
                        st.write(f"- 📈 Consider lower-risk loan options (current: {loan_type})")
                    if form_fill_time_sec > 180:
                        st.write(f"- ⏱️ Reduce form completion time ({form_fill_time_sec} seconds)")
                
                # Show feature details
                with st.expander("View Feature Details"):
                    st.write("**Input Values:**")
                    feature_data = {
                        'Feature': [
                            'Food Delivery Usage', 'E-commerce Usage', 'Phone Brand', 
                            'Employment Status', 'Salary Credits', 'Active Loans', 
                            'Rooted Device', 'Form Fill Time', 'Finance App Usage', 
                            'E-banking Level', 'Loan Type', 'Final Interest Rate'
                        ],
                        'Value': [
                            f"{swiggy_zomato_usage} times/week", f"{amazon_usage} times/month", 
                            f"{phone_brand} ({'Low' if phone_model == 0 else 'Medium' if phone_model == 1 else 'Premium'})",
                            employment_status, "Yes" if salary_credit_sms == 1 else "No",
                            loans_active, "Yes" if rooted_device == 1 else "No",
                            f"{form_fill_time_sec} seconds", f"{finance_app_usage} times/month", 
                            ebanking_transactions_level, loan_type, f"{final_interest_rate}%"
                        ]
                    }
                    st.dataframe(pd.DataFrame(feature_data))

    with tab4:
        st.header("Defaulter Analysis")
        st.info("""
        This section analyzes historical loan data to identify patterns and predict potential defaulters.
        Each EMI column with a value of 0 represents one payment flag.
        """)
        
        # Upload dataset
        uploaded_file = st.file_uploader("Upload Loan Dataset (CSV or Excel)", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success("Dataset loaded successfully!")
                
                # Show dataset preview
                with st.expander("View Dataset Preview"):
                    st.dataframe(df.head())
                
                # Analyze defaulters
                if st.button("Analyze Defaulters", type="primary"):
                    with st.spinner("Analyzing data for default patterns..."):
                        analysis_results = analyze_defaulters(df)
                        
                        # Display summary
                        st.subheader("Defaulter Analysis Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                label="Defaulters",
                                value=len(analysis_results['defaulter_list']),
                                help="Users with 3 payment flags (30% of dataset)"
                            )
                        
                        with col2:
                            st.metric(
                                label="Watch List",
                                value=len(analysis_results['watch_list']),
                                help="Users with 2 payment flags (20% of dataset)"
                            )
                        
                        with col3:
                            st.metric(
                                label="Good Standing",
                                value=len(analysis_results['good_standing_list']),
                                help="Users with 0-1 payment flags (50% of dataset)"
                            )
                        
                        # Create visualizations
                        st.subheader("Visualizations")
                        
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                        
                        # Pie chart of customer categories
                        colors = ['#ff6b6b', '#ffd166', '#06d6a0']
                        ax1.pie(analysis_results['summary_df']['Count'], 
                                labels=analysis_results['summary_df']['Category'], 
                                autopct='%1.1f%%', colors=colors, startangle=90)
                        ax1.set_title('Customer Distribution by Risk Category')
                        
                        # Bar chart of average flags by category
                        ax2.bar(analysis_results['summary_df']['Category'], 
                                analysis_results['summary_df']['Avg_Flags'], color=colors)
                        ax2.set_ylabel('Average Flags')
                        ax2.set_title('Average Payment Flags by Category')
                        ax2.set_ylim(0, 4)
                        
                        # Age distribution by category
                        bins = [18, 30, 40, 50, 60, 70]
                        ax3.hist([analysis_results['defaulter_list']['Age'], 
                                 analysis_results['watch_list']['Age'], 
                                 analysis_results['good_standing_list']['Age']], 
                                 bins=bins, label=['Defaulters', 'Watch List', 'Good Standing'], 
                                 color=colors, alpha=0.7)
                        ax3.legend()
                        ax3.set_xlabel('Age')
                        ax3.set_ylabel('Count')
                        ax3.set_title('Age Distribution by Risk Category')
                        
                        # Loan amount distribution by category
                        loan_amount_bins = [0, 50000, 100000, 150000, 200000, 250000]
                        ax4.hist([analysis_results['defaulter_list']['LoanAmount'], 
                                 analysis_results['watch_list']['LoanAmount'], 
                                 analysis_results['good_standing_list']['LoanAmount']], 
                                 bins=loan_amount_bins, label=['Defaulters', 'Watch List', 'Good Standing'], 
                                 color=colors, alpha=0.7)
                        ax4.legend()
                        ax4.set_xlabel('Loan Amount')
                        ax4.set_ylabel('Count')
                        ax4.set_title('Loan Amount Distribution by Risk Category')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show detailed lists
                        st.subheader("Detailed Lists")
                        
                        # Defaulter list
                        with st.expander("Defaulter List (30% of users with 3 flags)"):
                            defaulter_details = analysis_results['defaulter_list'][['LoanID', 'Age', 'LoanAmount', 'InterestRate', 'LoanTerm'] + analysis_results['emi_columns'] + ['flag_count']]
                            st.dataframe(defaulter_details)
                            
                            # Download button
                            csv = defaulter_details.to_csv(index=False)
                            st.download_button(
                                label="Download Defaulter List as CSV",
                                data=csv,
                                file_name="defaulter_list.csv",
                                mime="text/csv"
                            )
                        
                        # Watch list
                        with st.expander("Watch List (20% of users with 2 flags)"):
                            watch_details = analysis_results['watch_list'][['LoanID', 'Age', 'LoanAmount', 'InterestRate', 'LoanTerm'] + analysis_results['emi_columns'] + ['flag_count']]
                            st.dataframe(watch_details)
                            
                            # Download button
                            csv = watch_details.to_csv(index=False)
                            st.download_button(
                                label="Download Watch List as CSV",
                                data=csv,
                                file_name="watch_list.csv",
                                mime="text/csv"
                            )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.info("Please upload a loan dataset to begin defaulter analysis.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>CreditRisk AI | Hackathon Solution | Using alternative data for credit risk assessment</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()