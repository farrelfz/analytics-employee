import pandas as pd
from typing import Dict, Any, List

# Define the exact feature names and order as expected by the models
FEATURE_NAMES: List[str] = [
    'Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
    'PercentSalaryHike', 'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsSinceLastPromotion',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'Department_Research & Development', 'Department_Sales',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree', 'Gender_Male',
    'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative'
]

def build_feature_vector(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms raw input dictionary into a 1-row DataFrame containing 
    the 37 preprocessed features in the exact order expected by the models.
    """
    # Initialize all features with 0
    vector = {feat: 0.0 for feat in FEATURE_NAMES}
    
    # 1. Fill in numerical features (and ordinal features)
    numerical_cols = [
        'Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'JobInvolvement', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
        'PercentSalaryHike', 'RelationshipSatisfaction', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsSinceLastPromotion'
    ]
    for col in numerical_cols:
        vector[col] = float(input_data.get(col, 0))
        
    # 2. Fill in categorical features (one-hot encoded, with drop-first baselines)
    
    # BusinessTravel: Reference is 'Non-Travel'
    bt = input_data.get('BusinessTravel', 'Non-Travel')
    if bt == 'Travel_Frequently':
        vector['BusinessTravel_Travel_Frequently'] = 1.0
    elif bt == 'Travel_Rarely':
        vector['BusinessTravel_Travel_Rarely'] = 1.0
        
    # Department: Reference is 'Human Resources'
    dept = input_data.get('Department', 'Human Resources')
    if dept == 'Research & Development':
        vector['Department_Research & Development'] = 1.0
    elif dept == 'Sales':
        vector['Department_Sales'] = 1.0
        
    # EducationField: Reference is 'Human Resources'
    ef = input_data.get('EducationField', 'Human Resources')
    if ef in ['Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree']:
        vector[f'EducationField_{ef}'] = 1.0
        
    # Gender: Reference is 'Female'
    gender = input_data.get('Gender', 'Female')
    if gender == 'Male':
        vector['Gender_Male'] = 1.0
        
    # MaritalStatus: Reference is 'Divorced'
    ms = input_data.get('MaritalStatus', 'Divorced')
    if ms in ['Married', 'Single']:
        vector[f'MaritalStatus_{ms}'] = 1.0
        
    # OverTime: Reference is 'No'
    ot = input_data.get('OverTime', 'No')
    if ot == 'Yes':
        vector['OverTime_Yes'] = 1.0
        
    # JobRole: Reference is 'Healthcare Representative'
    jr = input_data.get('JobRole', 'Healthcare Representative')
    job_roles_in_features = [
        'Human Resources', 'Laboratory Technician', 'Manager',
        'Manufacturing Director', 'Research Director', 'Research Scientist',
        'Sales Executive', 'Sales Representative'
    ]
    if jr in job_roles_in_features:
        vector[f'JobRole_{jr}'] = 1.0
        
    # Create DataFrame and ensure columns are ordered exactly as expected
    df = pd.DataFrame([vector])
    return df[FEATURE_NAMES]
