import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AuraSeg - Premium Customer Segmentation Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Configurations ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "Mall_Customers.csv")

# --- Load Models & Data ---
@st.cache_resource
def load_models():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

@st.cache_data
def load_baseline_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

model, scaler = load_models()
df_base = load_baseline_data()

# --- App Constants ---
SEGMENTS = {
    0: {
        "name": "Middle of the Road",
        "color": "#a855f7",
        "bgColor": "rgba(168, 85, 247, 0.15)",
        "desc": "Moderate annual income and moderate spending habits. They represent the largest customer demographic group and are generally stable and brand-neutral.",
        "avgIncome": "₹45.9L",
        "avgSpending": "49.5",
        "strategy": "Engage with classic loyalty programs, milestone-based reward coupons, and mid-tier product recommendations."
    },
    1: {
        "name": "High-Value Targets (Star)",
        "color": "#10b981",
        "bgColor": "rgba(16, 185, 129, 0.15)",
        "desc": "High annual income paired with high spending score. Mostly young to middle-aged adults seeking premium boutique items, exclusive rewards, and direct service.",
        "avgIncome": "₹71.8L",
        "avgSpending": "82.1",
        "strategy": "Target with early access to luxury collections, invite-only VIP events, customized high-end styling, and premium product lines."
    },
    2: {
        "name": "Trendsetting Spendthrifts",
        "color": "#3b82f6",
        "bgColor": "rgba(59, 130, 246, 0.15)",
        "desc": "Low annual income but very high spending score. Primarily composed of younger consumers who prioritize modern fashion trends, social proof, and buy-now-pay-later financing.",
        "avgIncome": "₹21.3L",
        "avgSpending": "79.4",
        "strategy": "Deploy dynamic social media marketing, short-term flash discounts, trendy budget accessories, and seamless digital checkout options."
    },
    3: {
        "name": "Careful Skeptics",
        "color": "#f59e0b",
        "bgColor": "rgba(245, 158, 11, 0.15)",
        "desc": "High annual income combined with low spending score. Methodical shoppers who focus on quality, return on investment, and product utility. Mostly male demographically.",
        "avgIncome": "₹73.2L",
        "avgSpending": "17.1",
        "strategy": "Focus on durability, utility, detailed feature lists, and long-term value. Offer trial periods, extensions on warranties, and high-quality customer service."
    },
    4: {
        "name": "Frugal Conservatives",
        "color": "#ef4444",
        "bgColor": "rgba(239, 68, 68, 0.15)",
        "desc": "Low income matching low spending scores. Older customer demographic who prioritize essentials, budget-friendliness, and practical products.",
        "avgIncome": "₹21.8L",
        "avgSpending": "20.9",
        "strategy": "Offer everyday lowest-price guarantees, bulk discounts, essential product categories, and simple, zero-friction loyalty programs."
    }
}

# --- Helper Functions ---
def clean_html(html_str):
    """Strips leading and trailing spaces from each line of HTML to ensure markdown engine parses it as HTML (not code)."""
    return "\n".join([line.strip() for line in html_str.split("\n")])

def format_rupee_compact(income_in_k):
    val = income_in_k * 83000
    if val >= 10000000:
        return f"₹{val / 10000000:.1f} Cr"
    elif val >= 100000:
        return f"₹{val / 100000:.1f} L"
    else:
        return f"₹{val:,.0f}"

def get_income_column(df_cols):
    income_keys = ["annual income (k$)", "annual income (k₹)", "annual income (lakhs)", "annual income (inr)", "annual income", "income"]
    for col in df_cols:
        if col.lower() in income_keys or any(k in col.lower() for k in ["annual income", "income"]):
            return col
    return None

def get_spending_column(df_cols):
    spending_keys = ["spending score (1-100)", "spending score", "spending"]
    for col in df_cols:
        if col.lower() in spending_keys or any(k in col.lower() for k in ["spending score", "spending"]):
            return col
    return None

def normalize_income(val, col_name):
    if pd.isna(val):
        return 60.0
    try:
        if isinstance(val, str):
            val = val.replace("₹", "").replace("$", "").replace(",", "").strip()
        val = float(val)
    except ValueError:
        return 60.0
    
    if val > 1000:
        return val / 83000
    if col_name and "lakh" in col_name.lower():
        return val / 0.83
    return val

# Pre-calculate base dataset clusters
if df_base is not None and model is not None and scaler is not None:
    X_base = df_base[['Annual Income (k$)', 'Spending Score (1-100)']]
    df_base['Segment'] = model.predict(scaler.transform(X_base))

# --- CSS Styling Injection ---
st.markdown(clean_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Outfit', 'Inter', sans-serif;
    background-color: #0b0f19;
    color: #f3f4f6;
}

/* Sidebar overrides */
[data-testid="stSidebar"] {
    background-color: #0d1222;
    border-right: 1px solid #1e293b;
}

/* Custom cards styling */
.glass-card {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.card-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
}

.card-subtitle {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Header UI Styling */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 20px;
    margin-bottom: 30px;
}

.logo-text h1 {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.logo-text span {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: #a855f7;
    text-transform: uppercase;
}

/* Glow animation for status indicator */
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.status-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s infinite;
}

/* Tab button overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid #1e293b;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #94a3b8;
    padding: 10px 20px;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    transition: all 0.2s ease-in-out;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff;
    background-color: rgba(255, 255, 255, 0.05);
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background-color: rgba(168, 85, 247, 0.1) !important;
    border-bottom: 2px solid #a855f7 !important;
}

/* Glow animations for the live output cards */
@keyframes borderGlow0 {
    0% { border-color: rgba(168, 85, 247, 0.3); box-shadow: 0 0 8px rgba(168, 85, 247, 0.1); }
    100% { border-color: rgba(168, 85, 247, 0.9); box-shadow: 0 0 25px rgba(168, 85, 247, 0.5); }
}
@keyframes borderGlow1 {
    0% { border-color: rgba(16, 185, 129, 0.3); box-shadow: 0 0 8px rgba(16, 185, 129, 0.1); }
    100% { border-color: rgba(16, 185, 129, 0.9); box-shadow: 0 0 25px rgba(16, 185, 129, 0.5); }
}
@keyframes borderGlow2 {
    0% { border-color: rgba(59, 130, 246, 0.3); box-shadow: 0 0 8px rgba(59, 130, 246, 0.1); }
    100% { border-color: rgba(59, 130, 246, 0.9); box-shadow: 0 0 25px rgba(59, 130, 246, 0.5); }
}
@keyframes borderGlow3 {
    0% { border-color: rgba(245, 158, 11, 0.3); box-shadow: 0 0 8px rgba(245, 158, 11, 0.1); }
    100% { border-color: rgba(245, 158, 11, 0.9); box-shadow: 0 0 25px rgba(245, 158, 11, 0.5); }
}
@keyframes borderGlow4 {
    0% { border-color: rgba(239, 68, 68, 0.3); box-shadow: 0 0 8px rgba(239, 68, 68, 0.1); }
    100% { border-color: rgba(239, 68, 68, 0.9); box-shadow: 0 0 25px rgba(239, 68, 68, 0.5); }
}

.glow-card-0 { animation: borderGlow0 3s infinite alternate ease-in-out; }
.glow-card-1 { animation: borderGlow1 3s infinite alternate ease-in-out; }
.glow-card-2 { animation: borderGlow2 3s infinite alternate ease-in-out; }
.glow-card-3 { animation: borderGlow3 3s infinite alternate ease-in-out; }
.glow-card-4 { animation: borderGlow4 3s infinite alternate ease-in-out; }

/* Segment profile hover interactions */
.segment-profile-card {
    transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
}
.segment-profile-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.45);
    background: rgba(21, 32, 54, 0.8) !important;
}

h2, h3, h4 {
    font-family: 'Outfit', sans-serif;
}
</style>
"""), unsafe_allow_html=True)

# --- Header Section ---
st.markdown(clean_html("""
<div class="app-header">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.6rem; color: white; font-family: 'Outfit';">
            A
        </div>
        <div class="logo-text">
            <h1 style="font-size: 1.8rem; line-height: 1.1; margin: 0; font-family: 'Outfit';">AuraSeg</h1>
            <span style="font-size: 0.72rem; font-weight: 600; letter-spacing: 1.5px; color: #a855f7;">AI Customer Intelligence</span>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 20px; padding: 6px 16px;">
        <span class="status-dot"></span>
        <span style="font-size: 0.82rem; color: #10b981; font-weight: 500;">K-Means Engine: Online</span>
    </div>
</div>
"""), unsafe_allow_html=True)

# --- Sidebar: Live Profiler ---
st.sidebar.markdown(clean_html("""
<div style="margin-bottom: 20px;">
    <h2 style="font-family: 'Outfit', sans-serif; font-size: 1.4rem; color: white; margin: 0 0 5px 0;">Live Profiler</h2>
    <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">Adjust parameters to predict customer segment instantly.</p>
</div>
"""), unsafe_allow_html=True)

# Sidebar Inputs
gender = st.sidebar.radio("Customer Gender", ["Female", "Male"], index=0)
age = st.sidebar.slider("Age", min_value=18, max_value=70, value=30)
income = st.sidebar.slider("Annual Income (k$ index)", min_value=15, max_value=137, value=60, help="Annual Income parameter for K-Means. Real-time conversion to Lakhs is displayed below.")
spending = st.sidebar.slider("Spending Score (1-100)", min_value=1, max_value=99, value=50)

# Real-time Rupee conversion display
st.sidebar.markdown(clean_html(f"""
<div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 10px; margin: 15px 0; border: 1px solid rgba(255,255,255,0.05); font-size: 0.85rem;">
    <div style="display:flex; justify-content:space-between; margin-bottom: 4px;">
        <span style="color: #94a3b8;">Scaled Income Profile:</span>
        <span style="color: #ffffff; font-weight: 600;">{format_rupee_compact(income)}</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
        <span style="color: #94a3b8;">Spending Profile:</span>
        <span style="color: #ffffff; font-weight: 600;">{spending} / 100</span>
    </div>
</div>
"""), unsafe_allow_html=True)

# Run Live Prediction
cluster_id = 0
segment = SEGMENTS[0]
if model is not None and scaler is not None:
    features_live = pd.DataFrame([[income, spending]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
    scaled_input = scaler.transform(features_live)
    cluster_id = int(model.predict(scaled_input)[0])
    segment = SEGMENTS[cluster_id]

# Sidebar Output Card
st.sidebar.markdown(clean_html(f"""
<div class="glow-card-{cluster_id}" style="background: rgba(15, 23, 42, 0.95); border-radius: 14px; border: 1px solid {segment['color']}; border-top: 6px solid {segment['color']}; padding: 18px; margin-top: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <span style="background: {segment['bgColor']}; color: {segment['color']}; font-weight: 600; padding: 4px 10px; border-radius: 8px; font-size: 0.8rem;">
            Segment {cluster_id}
        </span>
        <span style="color: #64748b; font-size: 0.75rem; font-weight: 600; text-transform: uppercase;">Result</span>
    </div>
    <h3 style="margin: 0 0 8px 0; font-family: 'Outfit', sans-serif; font-size: 1.25rem; color: #ffffff;">{segment['name']}</h3>
    <p style="margin: 0 0 16px 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">{segment['desc']}</p>
    
    <div style="display: flex; gap: 8px; margin-bottom: 16px;">
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 8px 6px; flex: 1; text-align: center;">
            <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Centroid Income</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #ffffff; margin-top: 4px;">{segment['avgIncome']}</div>
        </div>
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 8px 6px; flex: 1; text-align: center;">
            <div style="font-size: 0.65rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Centroid Spend</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #ffffff; margin-top: 4px;">{segment['avgSpending']}</div>
        </div>
    </div>
    
    <div style="border-top: 1px solid #1e293b; padding-top: 12px;">
        <div style="font-size: 0.8rem; font-weight: 700; color: {segment['color']}; margin-bottom: 6px; display: flex; align-items: center; gap: 4px;">
            🎯 Engagement Strategy
        </div>
        <p style="margin: 0; font-size: 0.8rem; color: #cbd5e1; line-height: 1.45;">{segment['strategy']}</p>
    </div>
</div>
"""), unsafe_allow_html=True)


# --- Tabs Setup ---
tabs = st.tabs([
    "📈 Interactive Visualizer", 
    "📤 Batch CSV Upload", 
    "👥 Segment Profiles", 
    "🗃️ Customer Database"
])

# ==========================================
# TAB 1: INTERACTIVE VISUALIZER
# ==========================================
with tabs[0]:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="card-title">Customer Clustering Map</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Interactive 2D projection on Annual Income vs. Spending Score</div>', unsafe_allow_html=True)
    with col2:
        show_centroids = st.checkbox("Show Cluster Centroids", value=True)
        
    if df_base is not None and model is not None and scaler is not None:
        fig = go.Figure()
        
        # Add scatter traces for each segment
        for cid in sorted(df_base['Segment'].unique()):
            sub_df = df_base[df_base['Segment'] == cid]
            fig.add_trace(go.Scatter(
                x=sub_df['Annual Income (k$)'],
                y=sub_df['Spending Score (1-100)'],
                mode='markers',
                marker=dict(
                    size=9,
                    color=SEGMENTS[cid]['color'],
                    line=dict(width=1, color='rgba(0, 0, 0, 0.4)')
                ),
                name=SEGMENTS[cid]['name'],
                text=[
                    f"<b>ID:</b> {row['CustomerID']}<br>"
                    f"<b>Gender:</b> {row['Gender']}<br>"
                    f"<b>Age:</b> {row['Age']}<br>"
                    f"<b>Income:</b> {format_rupee_compact(row['Annual Income (k$)'])} ({row['Annual Income (k$)']}k$)<br>"
                    f"<b>Spending Score:</b> {row['Spending Score (1-100)']}"
                    for _, row in sub_df.iterrows()
                ],
                hoverinfo='text'
            ))
            
        # Add centroids if checked
        if show_centroids:
            scaled_centroids = model.cluster_centers_
            unscaled_centroids = scaler.inverse_transform(scaled_centroids)
            
            fig.add_trace(go.Scatter(
                x=unscaled_centroids[:, 0],
                y=unscaled_centroids[:, 1],
                mode='markers',
                marker=dict(
                    size=16,
                    symbol='x-thin',
                    color='#ffffff',
                    line=dict(width=3, color='#ffffff')
                ),
                name='Segment Centroid',
                text=[f"<b>Centroid {i}</b><br>Income: {format_rupee_compact(unscaled_centroids[i, 0])}<br>Spending Score: {unscaled_centroids[i, 1]:.1f}" for i in range(5)],
                hoverinfo='text'
            ))
            
        # Add active profile tracker
        fig.add_trace(go.Scatter(
            x=[income],
            y=[spending],
            mode='markers',
            marker=dict(
                size=18,
                symbol='diamond',
                color=segment['color'],
                line=dict(width=2, color='#ffffff')
            ),
            name='Active Profile',
            text=[f"<b>Active Profile</b><br>Income: {format_rupee_compact(income)} (${income}k)<br>Spending Score: {spending}"],
            hoverinfo='text'
        ))
        
        # Configure layout to be clean and modern
        fig.update_layout(
            xaxis=dict(
                title="Annual Income (k$)",
                gridcolor="#1e293b",
                zeroline=False,
                range=[10, 145]
            ),
            yaxis=dict(
                title="Spending Score (1-100)",
                gridcolor="#1e293b",
                zeroline=False,
                range=[0, 105]
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(color="#94a3b8")
            ),
            font=dict(color="#f3f4f6", family="Outfit"),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Visualizer cannot load because the baseline dataset is missing.")

# ==========================================
# TAB 2: BATCH CSV UPLOAD
# ==========================================
with tabs[1]:
    st.markdown('<div class="card-title">Upload Customer Directory</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Provide a CSV file with Annual Income and Spending Score values to classify them en masse.</div>', unsafe_allow_html=True)
    
    col_u1, col_u2 = st.columns([2, 1])
    
    with col_u1:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")
    with col_u2:
        # Create CSV template for download
        template_df = pd.DataFrame({
            'CustomerID': [1001, 1002, 1003],
            'Gender': ['Male', 'Female', 'Female'],
            'Age': [29, 43, 22],
            'Annual Income (k$)': [65, 88, 18],
            'Spending Score (1-100)': [48, 85, 76]
        })
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Template.csv",
            data=csv_template,
            file_name="template.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
            
            # Map columns
            income_col = get_income_column(up_df.columns)
            spending_col = get_spending_column(up_df.columns)
            
            if not income_col or not spending_col:
                st.error("Error: Could not identify 'Annual Income' or 'Spending Score' columns. Please check that your columns match the template.")
            else:
                # Preprocess uploaded dataset
                norm_income = up_df[income_col].apply(lambda x: normalize_income(x, income_col))
                norm_spending = pd.to_numeric(up_df[spending_col], errors='coerce').fillna(50.0)
                
                # Scale & Predict
                features = pd.DataFrame({
                    'Annual Income (k$)': norm_income,
                    'Spending Score (1-100)': norm_spending
                })
                scaled_features = scaler.transform(features)
                up_df['SegmentID'] = model.predict(scaled_features)
                
                # Display Results Dashboard - Using container with border for custom glassmorphic styling
                with st.container(border=True):
                    st.markdown('<h3 style="color:#ffffff; margin-bottom: 15px;">Batch Analysis Results</h3>', unsafe_allow_html=True)
                    
                    # Metric Cards
                    total_classified = len(up_df)
                    dominant_segment_id = int(up_df['SegmentID'].mode()[0])
                    dominant_segment_name = SEGMENTS[dominant_segment_id]['name']
                    avg_spending = float(norm_spending.mean())
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Classified", f"{total_classified} Customers")
                    m2.metric("Dominant Segment", f"Segment {dominant_segment_id}", f"{dominant_segment_name}")
                    m3.metric("Avg. Spending Score", f"{avg_spending:.1f} / 100")
                    
                    st.markdown("---")
                    
                    # Graphics & Data layout
                    d_col1, d_col2 = st.columns([1, 1])
                    
                    with d_col1:
                        # Pie chart
                        counts = up_df['SegmentID'].value_counts().reset_index()
                        counts.columns = ['SegmentID', 'Count']
                        counts['SegmentName'] = counts['SegmentID'].map(lambda x: SEGMENTS[x]['name'])
                        
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=counts['SegmentName'],
                            values=counts['Count'],
                            marker=dict(colors=[SEGMENTS[x]['color'] for x in counts['SegmentID']]),
                            hole=0.45,
                            textinfo='percent+label',
                            hoverinfo='label+value'
                        )])
                        
                        fig_pie.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color="#f3f4f6", family="Outfit"),
                            margin=dict(l=0, r=0, t=20, b=0),
                            showlegend=False,
                            height=350
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    with d_col2:
                        # Table preview
                        st.markdown("##### Preview (First 10 Classified Customers)")
                        preview_df = up_df.copy()
                        preview_df['Assigned Segment'] = preview_df['SegmentID'].map(lambda x: f"Segment {x}: {SEGMENTS[x]['name']}")
                        if 'CustomerID' in preview_df.columns:
                            cols_to_show = ['CustomerID', income_col, spending_col, 'Assigned Segment']
                        else:
                            cols_to_show = [income_col, spending_col, 'Assigned Segment']
                        st.dataframe(preview_df[cols_to_show].head(10), use_container_width=True, height=280)
                    
                    # Export Button
                    export_df = up_df.copy()
                    export_df['Segment'] = export_df['SegmentID']
                    export_df['Segment Name'] = export_df['SegmentID'].map(lambda x: SEGMENTS[x]['name'])
                    export_df = export_df.drop(columns=['SegmentID'])
                    
                    csv_data = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📤 Export Segmented CSV File",
                        data=csv_data,
                        file_name=f"segmented_{uploaded_file.name}",
                        mime="text/csv",
                        type="primary"
                    )
                
        except Exception as e:
            st.error(f"Error parsing batch CSV file: {e}")
    else:
        st.info("Please upload a CSV file to classify multiple customers at once.")

# ==========================================
# TAB 3: SEGMENT PROFILES
# ==========================================
with tabs[2]:
    st.markdown('<div class="card-title">Segment Profiles Directory</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">In-depth summary, statistical profiles, and action strategies for the 5 K-Means clusters.</div>', unsafe_allow_html=True)
    
    # Grid details function (Wrapped inside clean_html to prevent code block parsing)
    def render_segment_profile_card(cid, info):
        stats = {
            0: {"share": "40.5% (81/200)", "income": "₹45.9 Lakhs", "spending": "49.5 / 100", "age": "18 – 70 Years (Avg: 42.7)", "strategies": [
                "Offer standard loyalty points and subscription benefits.",
                "Incentivize up-spending with milestone-based discount coupons.",
                "Promote mid-tier product lines and everyday reliability."
            ]},
            1: {"share": "19.5% (39/200)", "income": "₹71.8 Lakhs", "spending": "82.1 / 100", "age": "27 – 40 Years (Avg: 32.7)", "strategies": [
                "Invite to private collection previews and VIP shopping lounges.",
                "Assign personal styling assistants and direct executive hotlines.",
                "Deliver high-end, premium brand collaboration opportunities."
            ]},
            2: {"share": "11.0% (22/200)", "income": "₹21.3 Lakhs", "spending": "79.4 / 100", "age": "18 – 35 Years (Avg: 25.3)", "strategies": [
                "Promote using dynamic social media visual media campaigns.",
                "Utilize flash-discount models and time-sensitive sales.",
                "Offer convenient digital financing options at checkout."
            ]},
            3: {"share": "17.5% (35/200)", "income": "₹73.2 Lakhs", "spending": "17.1 / 100", "age": "19 – 59 Years (Avg: 41.1)", "strategies": [
                "Emphasize premium specifications, durability, and craftsmanship.",
                "Provide high-fidelity specifications and side-by-side data tables.",
                "Offer extended warranties, trial periods, and satisfaction guarantees."
            ]},
            4: {"share": "11.5% (23/200)", "income": "₹21.8 Lakhs", "spending": "20.9 / 100", "age": "19 – 67 Years (Avg: 45.2)", "strategies": [
                "Ensure everyday lowest-price guarantees on core items.",
                "Promote bulk bundling discounts and high utility items.",
                "Deliver direct mailers and simple, zero-friction loyalty programs."
            ]}
        }
        
        s_stats = stats[cid]
        color = info['color']
        bg_color = info['bgColor']
        
        strategies_html = "".join([f"<li style='margin-bottom: 6px;'>{item}</li>" for item in s_stats['strategies']])
        
        return clean_html(f"""
        <div class="segment-profile-card" style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1e293b; border-top: 5px solid {color}; border-radius: 12px; padding: 20px; margin-bottom: 20px; height: 100%; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                <span style="background: {bg_color}; color: {color}; font-weight: 600; padding: 4px 10px; border-radius: 8px; font-size: 0.8rem;">
                    Segment {cid}
                </span>
                <strong style="color: #ffffff; font-size: 0.95rem; font-family: 'Outfit';">{info['name']}</strong>
            </div>
            <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.45; margin-bottom: 16px; min-height: 55px;">
                {info['desc']}
            </p>
            <div style="font-size: 0.82rem; border-top: 1px solid #1e293b; border-bottom: 1px solid #1e293b; padding: 12px 0; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">Segment Share</span>
                    <span style="color: #f8fafc; font-weight: 600;">{s_stats['share']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">Avg. Annual Income</span>
                    <span style="color: #f8fafc; font-weight: 600;">{s_stats['income']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #64748b;">Avg. Spending Score</span>
                    <span style="color: #f8fafc; font-weight: 600;">{s_stats['spending']}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #64748b;">Age Spectrum</span>
                    <span style="color: #f8fafc; font-weight: 600;">{s_stats['age']}</span>
                </div>
            </div>
            <div>
                <h4 style="font-size: 0.85rem; color: {color}; margin: 0 0 8px 0; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">🎯 Actionable Tactics</h4>
                <ul style="font-size: 0.8rem; color: #cbd5e1; margin: 0; padding-left: 16px; line-height: 1.4;">
                    {strategies_html}
                </ul>
            </div>
        </div>
        """)

    # Grid layout of profiles
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.markdown(render_segment_profile_card(0, SEGMENTS[0]), unsafe_allow_html=True)
    with row1_c2:
        st.markdown(render_segment_profile_card(1, SEGMENTS[1]), unsafe_allow_html=True)
        
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.markdown(render_segment_profile_card(2, SEGMENTS[2]), unsafe_allow_html=True)
    with row2_c2:
        st.markdown(render_segment_profile_card(3, SEGMENTS[3]), unsafe_allow_html=True)
        
    row3_c1, row3_c2 = st.columns(2)
    with row3_c1:
        st.markdown(render_segment_profile_card(4, SEGMENTS[4]), unsafe_allow_html=True)
    with row3_c2:
        # A nice dashboard summary panel in place of the empty spot - Wrapped in clean_html
        st.markdown(clean_html("""
        <div style="background: linear-gradient(135deg, rgba(168, 85, 247, 0.05) 0%, rgba(59, 130, 246, 0.05) 100%); border: 1px dashed #3b82f6; border-radius: 12px; padding: 20px; height: 100%; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h4 style="color:#ffffff; margin: 0 0 10px 0; font-family:'Outfit'; font-size:1.1rem; display:flex; align-items:center; gap:8px;">
                💡 Segmentation Methodology
            </h4>
            <p style="color:#94a3b8; font-size:0.82rem; margin:0 0 12px 0; line-height:1.45;">
                These profiles are computed using a K-Means clustering algorithm trained on normalized annual income and spending score profiles. Standardization maps the indices to zero-mean and unit variance, allowing Euclidean distance calculations to represent true profile differences without scales biasing results.
            </p>
            <div style="font-size:0.8rem; color:#a855f7; font-weight:600;">
                Optimal Cluster Count (k = 5) derived via Elbow Method & Silhouette Scores.
            </div>
        </div>
        """), unsafe_allow_html=True)

# ==========================================
# TAB 4: CUSTOMER DATABASE EXPLORER
# ==========================================
with tabs[3]:
    st.markdown('<div class="card-title">Customer Database Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subtitle">Search, filter, and review the base dataset of 200 mall customers.</div>', unsafe_allow_html=True)
    
    if df_base is not None:
        db_col1, db_col2 = st.columns([2, 1])
        with db_col1:
            search_query = st.text_input("🔍 Search customer database (by Customer ID or Gender)", placeholder="Search...")
        with db_col2:
            filter_seg = st.selectbox(
                "Filter by Segment",
                options=["All Segments"] + [f"Segment {i} ({SEGMENTS[i]['name']})" for i in range(5)]
            )
            
        # Filter Logic
        df_filtered = df_base.copy()
        
        # Text Search Filter
        if search_query:
            df_filtered = df_filtered[
                df_filtered['CustomerID'].astype(str).str.contains(search_query, case=False, na=False) |
                df_filtered['Gender'].str.contains(search_query, case=False, na=False)
            ]
            
        # Segment Filter
        if filter_seg != "All Segments":
            selected_seg_id = int(filter_seg.split(" ")[1])
            df_filtered = df_filtered[df_filtered['Segment'] == selected_seg_id]
            
        # Create user facing display dataframe
        df_display = df_filtered.copy()
        df_display['Assigned Segment'] = df_display['Segment'].map(lambda x: f"Segment {x}: {SEGMENTS[x]['name']}")
        df_display['Annual Income'] = df_display['Annual Income (k$)'].map(lambda x: f"₹{x*83000/100000:.1f}L (${x}k)")
        df_display = df_display.rename(columns={
            'CustomerID': 'Customer ID',
            'Spending Score (1-100)': 'Spending Score'
        })
        
        # Sort or display columns
        columns_order = ['Customer ID', 'Gender', 'Age', 'Annual Income', 'Spending Score', 'Assigned Segment']
        st.dataframe(
            df_display[columns_order], 
            use_container_width=True,
            height=450
        )
        
        # Stats summary for filters - Wrapped in clean_html
        st.markdown(clean_html(f"""
        <div style="font-size: 0.82rem; color: #64748b; text-align: right; margin-top: 5px;">
            Showing {len(df_filtered)} of {len(df_base)} entries
        </div>
        """), unsafe_allow_html=True)
    else:
        st.info("Customer database is not available because the dataset file is missing.")


# --- Footer ---
st.markdown(clean_html("""
<div style="border-top: 1px solid #1e293b; margin-top: 50px; padding-top: 15px; padding-bottom: 20px; display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #64748b;">
    <p style="margin: 0;">&copy; 2026 AuraSeg. Constructed for Customer Segmentation Analytics.</p>
    <div style="display: flex; gap: 15px;">
        <span>Model Version: KMeans v1.0</span>
        <span>Accuracy Metric (Silhouette): 0.553</span>
    </div>
</div>
"""), unsafe_allow_html=True)
