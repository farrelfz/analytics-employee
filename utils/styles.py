import streamlit as st

def inject_custom_css() -> None:
    """
    Injects a modern, clean, and visually pleasing UI design system.
    Uses Indigo and Slate colors with rounded card structures for a premium look.
    """
    st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        
        <style>
            /* Global variables & reset */
            :root {
                --primary: #4F46E5;        /* Indigo 600 */
                --primary-light: #818CF8;  /* Indigo 400 */
                --primary-dark: #3730A3;   /* Indigo 800 */
                --background: #FFFFFF;
                --text-main: #0F172A;      /* Slate 900 */
                --text-muted: #64748B;     /* Slate 500 */
                --card-bg: #F8FAFC;        /* Slate 50 */
                --border-color: #E2E8F0;   /* Slate 200 */
                
                --color-no: #3B82F6;       /* Royal Blue */
                --color-yes: #F43F5E;      /* Coral Red */
            }
            
            /* Typography override */
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Plus Jakarta Sans', sans-serif !important;
                color: #1E1B4B !important; /* Indigo 950 */
                font-weight: 700 !important;
                margin-top: 0.75rem !important;
                margin-bottom: 0.75rem !important;
                letter-spacing: -0.01em !important;
            }
            
            p, li, span, label, input, select, button {
                font-family: 'Inter', sans-serif !important;
                color: #334155; /* Slate 700 */
            }
            
            /* Elegant Title & Subtitle */
            .app-title {
                font-family: 'Plus Jakarta Sans', sans-serif;
                font-size: 2.2rem;
                font-weight: 800;
                background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.2rem;
                letter-spacing: -0.025em;
            }
            .app-subtitle {
                font-family: 'Inter', sans-serif;
                font-size: 1rem;
                color: #64748B;
                margin-bottom: 1.75rem;
                font-weight: 400;
            }
            
            /* Metric Grid */
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 1.25rem;
                margin-bottom: 2rem;
            }
            
            /* Premium, rounded metric card with subtle borders and shadows */
            .metric-card {
                background: #FFFFFF;
                border: 1px solid #E2E8F0 !important;
                padding: 1.25rem 1.5rem;
                border-radius: 12px;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
                border-top: 4px solid #6366F1 !important; /* Top colored bar */
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
            }
            .metric-card.alert-card {
                border-top: 4px solid #F43F5E !important; /* Coral Red top bar */
                background: #FFF5F5;
            }
            .metric-card.success-card {
                border-top: 4px solid #3B82F6 !important; /* Royal Blue top bar */
                background: #F0FDFF;
            }
            .metric-title {
                font-size: 0.75rem;
                color: #64748B;
                text-transform: uppercase;
                font-weight: 600;
                letter-spacing: 0.075em;
                margin-bottom: 0.3rem;
            }
            .metric-value {
                font-family: 'Plus Jakarta Sans', sans-serif;
                font-size: 1.8rem;
                font-weight: 700;
                color: #0F172A;
            }
            
            /* Card Containers */
            .content-container {
                background: #FFFFFF;
                border: 1px solid #F1F5F9 !important;
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
                margin-bottom: 1.5rem;
            }
            
            /* Highlight Info Box */
            .highlight-box {
                background: #F8FAFC;
                border-left: 4px solid #6366F1 !important;
                padding: 1.25rem;
                border-radius: 0 8px 8px 0;
                margin-top: 1rem;
                margin-bottom: 1rem;
                font-size: 0.9rem;
                color: #475569;
            }
            .highlight-box p {
                margin: 0 !important;
                line-height: 1.6;
            }
            
            /* Tab Styles */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
                border-bottom: 1px solid #E2E8F0;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 8px 8px 0 0;
                padding: 0.6rem 1.2rem;
                background-color: transparent;
                border: none;
                font-weight: 600;
                color: #64748B;
                font-size: 0.95rem;
            }
            .stTabs [aria-selected="true"] {
                background-color: transparent !important;
                color: #4F46E5 !important;
                border-bottom: 2px solid #4F46E5 !important;
            }
            
            /* Recommendation Cards */
            .plan-card {
                background: #F8FAFC;
                border-left: 4px solid #4F46E5 !important;
                padding: 1rem 1.25rem;
                border-radius: 0 8px 8px 0;
                margin-bottom: 0.75rem;
                box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
            }
            .plan-card-title {
                font-family: 'Plus Jakarta Sans', sans-serif;
                font-weight: 700;
                color: #1E1B4B;
                font-size: 0.95rem;
            }
            .plan-card-body {
                font-size: 0.85rem;
                color: #475569;
                margin-top: 0.25rem;
                line-height: 1.5;
            }
        </style>
    """, unsafe_allow_html=True)
