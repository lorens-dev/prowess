import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & UTILS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Inbox", layout="wide", page_icon="")

# Constants
FILES = {
    "transactions": "transactions.csv",
    "transactiondetails": "transactiondetails.csv",
    "products": "products.csv",
}

def clean_currency(x):
    """Helper to strip currency symbols and commas."""
    if isinstance(x, str):
        clean_str = x.replace('₱', '').replace('$', '').replace(',', '').strip()
        try:
            return float(clean_str)
        except ValueError:
            return 0.0
    return x

# -----------------------------------------------------------------------------
# 2. ETL PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_filter_data(start_h, end_h):
    """
    Loads real CSV data and filters transactions based on the selected Hour Range.
    """
    try:
        # Load with generic encoding for safety
        df_trans = pd.read_csv(FILES['transactions'], encoding='unicode_escape')
        df_td = pd.read_csv(FILES['transactiondetails'], encoding='unicode_escape')
        df_prod = pd.read_csv(FILES['products'], encoding='unicode_escape')

    except FileNotFoundError as e:
        st.error(f"❌ **File Not Found:** {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Error Loading Data:** {str(e)}")
        st.stop()

    # --- 1. Process Transactions ---
    # Parse Dates
    df_trans['tdate'] = pd.to_datetime(df_trans['tdate'], errors='coerce')
    df_trans.dropna(subset=['tdate'], inplace=True)

    # Clean Numeric Columns
    for col in ['gross', 'cogs', 'grossprofit']:
        if col in df_trans.columns:
            df_trans[col] = df_trans[col].apply(clean_currency)
            df_trans[col] = pd.to_numeric(df_trans[col], errors='coerce').fillna(0)

    # Filter Voids
    if 'isvoid' in df_trans.columns:
        df_trans['isvoid'] = pd.to_numeric(df_trans['isvoid'], errors='coerce').fillna(0)
        df_trans = df_trans[df_trans['isvoid'] == 0].copy()

    # Extract Hour & Date
    df_trans['hour'] = df_trans['tdate'].dt.hour
    df_trans['date_only'] = df_trans['tdate'].dt.date
    df_trans['day_of_week'] = df_trans['tdate'].dt.dayofweek
    df_trans['month'] = df_trans['tdate'].dt.month
    df_trans['is_weekend'] = df_trans['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # --- FILTER BY TIME RANGE ---
    # We only keep transactions that happened within the selected shift
    if start_h <= end_h:
        # Standard range (e.g., 9 AM to 5 PM)
        df_trans = df_trans[(df_trans['hour'] >= start_h) & (df_trans['hour'] <= end_h)]
    else:
        # Cross-midnight range (e.g., 10 PM to 2 AM)
        df_trans = df_trans[(df_trans['hour'] >= start_h) | (df_trans['hour'] <= end_h)]

    # --- 2. Process Details ---
    df_td['productid'] = pd.to_numeric(df_td['productid'], errors='coerce')
    df_prod['productid'] = pd.to_numeric(df_prod['productid'], errors='coerce')

    if 'sellingprice' in df_td.columns:
        df_td['sellingprice'] = df_td['sellingprice'].apply(clean_currency)
    if 'quantity' in df_td.columns:
        df_td['quantity'] = pd.to_numeric(df_td['quantity'], errors='coerce').fillna(0)

    # Join Products & Trans Details
    df_full = df_td.merge(df_prod, on='productid', how='left')
    
    # Filter Details to match the filtered Transactions (by TID)
    valid_tids = df_trans['tid'].unique()
    df_full = df_full[df_full['tid'].isin(valid_tids)]
    
    # Merge Date info back into Details
    df_full = df_full.merge(df_trans[['tid', 'date_only']], on='tid', how='inner')

    return df_trans, df_full

# -----------------------------------------------------------------------------
# 3. FORECASTING ENGINE
# -----------------------------------------------------------------------------
def train_model(df_trans):
    """Trains a model on the filtered shift data."""
    # Aggregation: Sum of sales PER DAY (within the selected hours)
    daily_sales = df_trans.groupby('date_only')['gross'].sum().reset_index()
    daily_sales['date_only'] = pd.to_datetime(daily_sales['date_only'])

    # Re-index to fill missing days with 0 (Closed days)
    if not daily_sales.empty:
        full_idx = pd.date_range(start=daily_sales['date_only'].min(), end=daily_sales['date_only'].max(), freq='D')
        daily_sales = daily_sales.set_index('date_only').reindex(full_idx, fill_value=0).reset_index()
        daily_sales.rename(columns={'index': 'date_only'}, inplace=True)

    # Features
    daily_sales['day_of_week'] = daily_sales['date_only'].dt.dayofweek
    daily_sales['is_weekend'] = daily_sales['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    daily_sales['month'] = daily_sales['date_only'].dt.month

    # Lags
    daily_sales['lag_1'] = daily_sales['gross'].shift(1)
    daily_sales['lag_7'] = daily_sales['gross'].shift(7)
    daily_sales['rolling_mean_7'] = daily_sales['gross'].rolling(window=7).mean()
    
    daily_sales.dropna(inplace=True)

    if len(daily_sales) < 10:
        return None, daily_sales

    X = daily_sales[['day_of_week', 'is_weekend', 'month', 'lag_1', 'lag_7', 'rolling_mean_7']]
    y = daily_sales['gross']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, daily_sales

def get_prediction(model, recent_data, target_date):
    if model is None or recent_data.empty:
        return 0.0

    target_dt = pd.to_datetime(target_date)
    # Features for target
    day_of_week = target_dt.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    month = target_dt.month
    
    last_row = recent_data.iloc[-1]
    lag_1 = last_row['gross']
    lag_7 = recent_data.iloc[-7]['gross'] if len(recent_data) >= 7 else last_row['gross']
    rolling_7 = recent_data['gross'].tail(7).mean()
    
    features = pd.DataFrame([[day_of_week, is_weekend, month, lag_1, lag_7, rolling_7]], 
                            columns=['day_of_week', 'is_weekend', 'month', 'lag_1', 'lag_7', 'rolling_mean_7'])
    
    return max(0, model.predict(features)[0])

# -----------------------------------------------------------------------------
# 4. INVENTORY LOGIC
# -----------------------------------------------------------------------------
def get_stock_recommendations(df_full, target_date):
    # Stats per product
    stats = df_full.groupby('productname').agg(
        total_qty=('quantity', 'sum'),
        avg_price=('sellingprice', 'mean'),
        days_active=('date_only', 'nunique')
    ).reset_index()
    
    stats['days_active'] = stats['days_active'].replace(0, 1)
    stats['velocity'] = stats['total_qty'] / stats['days_active'] # Avg Qty per Shift
    
    is_weekend = target_date.weekday() >= 5
    recommendations = []
    
    for _, row in stats.iterrows():
        # Score Logic
        score = 0
        if row['velocity'] > 5: score += 40
        elif row['velocity'] > 2: score += 20
        else: score += 5
        
        if row['avg_price'] > 200: score += 20
        
        if is_weekend: score += 20
        
        if score >= 60: priority = "High"
        elif score >= 40: priority = "Medium"
        else: priority = "Low"
        
        # Predicted Demand
        multiplier = 1.3 if is_weekend else 1.0
        pred_qty = int(np.ceil(row['velocity'] * multiplier))
        
        recommendations.append({
            "Product": row['productname'],
            "Est. Demand": pred_qty,
            "Priority": priority,
            "Avg/Shift": round(row['velocity'], 1),
            "Score": score
        })
        
    return pd.DataFrame(recommendations).sort_values(by="Score", ascending=False)

# -----------------------------------------------------------------------------
# 5. UI & MAIN
# -----------------------------------------------------------------------------
def main():
    # --- SIDEBAR ---
    # use_container_width=True replaces use_column_width=True
    st.sidebar.image("logo.png", use_container_width=True)

    
    today = datetime.now().date()
    target_date = st.sidebar.date_input("Forecast Date", today + timedelta(days=1))
    
    st.sidebar.subheader("Shift Timings (12H Format)")
    t_start = st.sidebar.time_input("Start Time", time(17, 0)) 
    t_end = st.sidebar.time_input("End Time", time(22, 0))

    st.sidebar.caption("💡 The forecast updates based on the selected time range.")

    # --- PROCESS DATA ---
    with st.spinner('Analyzing Shift Data...'):
        df_trans, df_full = load_and_filter_data(t_start.hour, t_end.hour)
        model, recent_data = train_model(df_trans)

    # --- MODE DETECTION: ACTUAL vs FORECAST ---
    # Check if we have ACTUAL data for the selected date
    
    # 1. Is there data in the filtered dataframe for this specific date?
    actual_data_row = df_trans[df_trans['date_only'] == target_date]
    has_actual_data = not actual_data_row.empty

    # 2. Are we in the future relative to our last record?
    last_record_date = df_trans['date_only'].max() if not df_trans.empty else today
    is_future = target_date > last_record_date

    # --- DASHBOARD UI ---
    st.title(f" Dashboard: {target_date.strftime('%A, %b %d')}")
    
    metric_rev = 0
    metric_count = 0
    is_forecast_mode = False

    if has_actual_data:
        # --- SHOW ACTUALS ---
        st.success(f" Displaying **ACTUAL** recorded data for {target_date}.")
        metric_rev = actual_data_row['gross'].sum()
        metric_count = len(actual_data_row)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actual Revenue", f"₱{metric_rev:,.2f}")
        c2.metric("Actual Orders", f"{metric_count}")
        c3.metric("COGS (Actual)", f"₱{actual_data_row['cogs'].sum():,.2f}")
        c4.metric("Gross Profit", f"₱{actual_data_row['grossprofit'].sum():,.2f}")

    elif is_future or (not has_actual_data):
        # --- SHOW FORECAST ---
        is_forecast_mode = True
        st.warning(f"⚠️ Future Date / No Records Found. Displaying **AI PREDICTION**.")
        
        metric_rev = get_prediction(model, recent_data, target_date)
        avg_ticket = df_trans['gross'].mean() if not df_trans.empty else 0
        metric_count = int(metric_rev / avg_ticket) if avg_ticket > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Revenue", f"₱{metric_rev:,.2f}", help="AI Estimate")
        c2.metric("Predicted Orders", f"{metric_count}", help="AI Estimate")
        c3.metric("Est. COGS (30%)", f"₱{metric_rev*0.3:,.2f}")
        c4.metric("Est. Gross Profit", f"₱{metric_rev*0.7:,.2f}")

    st.markdown("---")

    # Tabs
    tab1, tab2 = st.tabs(["📈 Revenue Trends", "🍲 Inventory Plan"])

    with tab1:
        st.subheader(f"Revenue Trend ({t_start.strftime('%I:%M %p')} - {t_end.strftime('%I:%M %p')})")
        if not recent_data.empty:
            fig = px.line(recent_data, x='date_only', y='gross', title="Historical Performance")
            
            # Only add the Red Forecast Dot if we are in Forecast Mode
            if is_forecast_mode:
                fig.add_trace(go.Scatter(
                    x=[target_date], y=[metric_rev],
                    mode='markers+text', name='Forecast',
                    marker=dict(color='red', size=12),
                    text=[f"₱{metric_rev:,.0f}"], textposition="top center"
                ))
            else:
                # Add a Green Dot for the Actual selected data
                fig.add_trace(go.Scatter(
                    x=[target_date], y=[metric_rev],
                    mode='markers', name='Selected Date',
                    marker=dict(color='green', size=12)
                ))
                
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data found for this specific time range.")

    with tab2:
        st.subheader("Recommended Stocking")
        if not df_full.empty:
            inv_df = get_stock_recommendations(df_full, target_date)
            if not inv_df.empty:
                def color_prio(val):
                    if val == "High": return "color: #2ecc71; font-weight: bold"
                    if val == "Medium": return "color: #f1c40f; font-weight: bold"
                    return "color: #e74c3c"
                
                st.dataframe(inv_df.style.map(color_prio, subset=['Priority']), use_container_width=True, hide_index=True)
            else:
                st.info("No sales details found for this range.")
        else:
            st.info("No product data available.")

if __name__ == "__main__":
    main()