import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit.components.v1 as components
import json
from datetime import datetime, timedelta
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# ===== PAGE CONFIG =====
st.set_page_config(
    layout="wide", 
    page_title="🚀 AI-Powered Sales Dashboard",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM STYLING =====
st.markdown("""
<style>
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --background-color: #0f1419;
        --text-color: #e0e0e0;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
        font-weight: 500;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
    
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ===== TITLE & DESCRIPTION =====
st.title("🚀 AI-Powered Sales & Profit Dashboard")
st.markdown("**📊 Advanced Analytics | 🤖 Gemini AI Agent | 📈 Predictive Forecasting | 💡 Smart Insights**")
st.markdown("Upload your Excel data or use our preloaded sample dataset to get started!")

# ===== HELPER FUNCTIONS =====

@st.cache_data
def convert_df_to_csv(df):
    """Convert DataFrame to CSV bytes"""
    return df.to_csv(index=False).encode('utf-8')


@st.cache_data
def load_sample_data():
    """Load preloaded sample dataset"""
    try:
        all_sheets = pd.read_excel("PowerBI_Test_Data.xlsx", sheet_name=None)
        df_sales = all_sheets.get("Sales")
        df_products = all_sheets.get("Products")
        df_customers = all_sheets.get("Customers")
        
        if df_sales is None or df_products is None or df_customers is None:
            return None
        
        return {
            "sales": df_sales,
            "products": df_products,
            "customers": df_customers
        }
    except Exception as e:
        st.warning(f"Could not load preloaded dataset: {e}")
        return None


@st.cache_data
def process_data(df_sales, df_products, df_customers):
    """Merge and process data with calculations"""
    try:
        df_merged = pd.merge(df_sales, df_products, on="ProductID", how="left")
        df_merged = pd.merge(df_merged, df_customers, on="CustomerID", how="left")

        df_merged['Date'] = pd.to_datetime(df_merged['Date'])
        df_merged['Total Sales'] = df_merged['Quantity'] * df_merged['Unit Price']
        df_merged['Total Cost'] = df_merged['Quantity'] * df_merged['unit Cost'] 
        df_merged['Profit'] = df_merged['Total Sales'] - df_merged['Total Cost']
        df_merged['Profit Margin %'] = (df_merged['Profit'] / df_merged['Total Sales'].replace(0, np.nan)) * 100
        df_merged['Month-Year'] = df_merged['Date'].dt.to_period('M').astype(str)
        
        required_cols = ['Date', 'Total Sales', 'Profit', 'Quantity', 'SaleID', 'CustomerID', 'Category', 'City', 'ProductName', 'CustomerName']
        for col in required_cols:
            if col not in df_merged.columns:
                st.error(f"Error: Missing required column '{col}' after merging.")
                return None
                
        return df_merged
    except KeyError as e:
        st.error(f"Error: Missing expected column {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return None


def summarize_data_for_gemini(df_filtered):
    """Generate comprehensive data summary for Gemini AI"""
    try:
        total_revenue = df_filtered['Total Sales'].sum()
        total_profit = df_filtered['Profit'].sum()
        profit_margin = (total_profit / total_revenue) * 100 if total_revenue != 0 else 0
        total_units = df_filtered['Quantity'].sum()
        total_orders = df_filtered['SaleID'].nunique()
        unique_customers = df_filtered['CustomerID'].nunique()
        
        top_products = df_filtered.groupby('ProductName', observed=True)['Total Sales'].sum().nlargest(5).to_dict()
        top_products_profit = df_filtered.groupby('ProductName', observed=True)['Profit'].sum().nlargest(3).to_dict()
        top_customers = df_filtered.groupby('CustomerName', observed=True)['Total Sales'].sum().nlargest(5).to_dict()
        sales_by_category = df_filtered.groupby('Category', observed=True)['Total Sales'].sum().to_dict()
        sales_by_city = df_filtered.groupby('City', observed=True)['Total Sales'].sum().nlargest(5).to_dict()
        profit_by_category = df_filtered.groupby('Category', observed=True)['Profit'].sum().to_dict()
        
        # Calculate trends
        monthly_trend = df_filtered.set_index('Date').resample('ME')[['Total Sales', 'Profit']].sum()
        trend_direction = "↑ INCREASING" if len(monthly_trend) > 1 and monthly_trend['Total Sales'].iloc[-1] > monthly_trend['Total Sales'].iloc[0] else "↓ DECREASING"

        summary = f"""
        === COMPREHENSIVE DATA SUMMARY FOR AI ANALYSIS ===
        
        REVENUE & PROFITABILITY:
        - Total Revenue: ₹{total_revenue:,.2f}
        - Total Profit: ₹{total_profit:,.2f}
        - Profit Margin: {profit_margin:.2f}%
        - Avg Order Value: ₹{total_revenue/total_orders:,.2f}
        - Avg Profit per Order: ₹{total_profit/total_orders:,.2f}
        
        OPERATIONAL METRICS:
        - Total Units Sold: {total_units:,.0f}
        - Total Orders: {total_orders:,.0f}
        - Unique Customers: {unique_customers:,.0f}
        - Avg Items per Order: {total_units/total_orders:.2f}
        - Sales Trend: {trend_direction}
        
        TOP PERFORMERS:
        Top 5 Products by Sales: {json.dumps(top_products, indent=2)}
        Top 3 Products by Profit: {json.dumps(top_products_profit, indent=2)}
        Top 5 Customers by Sales: {json.dumps(top_customers, indent=2)}
        
        CATEGORY & GEOGRAPHICAL BREAKDOWN:
        Sales by Category: {json.dumps(sales_by_category, indent=2)}
        Profit by Category: {json.dumps(profit_by_category, indent=2)}
        Top 5 Cities by Sales: {json.dumps(sales_by_city, indent=2)}
        """
        return summary
    except Exception as e:
        return f"Could not summarize data: {str(e)}"


def get_gemini_fetch_html(prompt: str, output_element_id: str, loading_text: str) -> str:
    """Generate HTML/JS for Gemini API calls"""
    safe_prompt = json.dumps(prompt)
    
    html_js_content = f"""
    <div id="{output_element_id}" style="border: 2px solid #4fc3f7; border-radius: 12px; padding: 20px; min-height: 250px; background: linear-gradient(135deg, #0f1419 0%, #1a1a2e 100%); color: #e0e0e0; font-family: 'Segoe UI', sans-serif; line-height: 1.6;">
        <p style="color: #4fc3f7; font-weight: bold;">⏳ {loading_text}</p>
    </div>

    <script>
    (async function() {{
        const prompt = {safe_prompt};
        const apiKey = "AIzaSyDW-aDLbt8CN2DrvwWtKPewdYVLf4VqBtM";
        const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${{apiKey}}`;
        const outputEl = document.getElementById({json.dumps(output_element_id)});

        async function fetchWithBackoff(url, options, maxRetries = 5, baseDelay = 1000) {{
            let attempt = 0;
            while (attempt < maxRetries) {{
                try {{
                    const response = await fetch(url, options);
                    if (response.ok) {{
                        return await response.json();
                    }} else if (response.status === 429 || response.status >= 500) {{
                        throw new Error(`Retryable error: ${{response.status}}`);
                    }} else {{
                        const errorData = await response.json();
                        throw new Error(`API Error: ${{response.status}}`);
                    }}
                }} catch (error) {{
                    if (attempt >= maxRetries - 1) {{
                        throw error;
                    }}
                }}
                attempt++;
                const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            }}
        }}

        async function fetchGeminiResponse() {{
            const payload = {{
                contents: [{{ 
                    parts: [{{ text: prompt }}] 
                }}],
                generationConfig: {{
                    temperature: 0.7,
                    topK: 40,
                    topP: 0.95,
                    maxOutputTokens: 2048
                }}
            }};

            try {{
                const result = await fetchWithBackoff(apiUrl, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload)
                }});
                
                if (result.candidates && result.candidates[0] && result.candidates[0].content && result.candidates[0].content.parts && result.candidates[0].content.parts[0] && result.candidates[0].content.parts[0].text) {{
                    let text = result.candidates[0].content.parts[0].text;
                    
                    text = text.replace(/\\*\\*\\*([^*]+)\\*\\*\\*/g, '<strong><em>$1</em></strong>');
                    text = text.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
                    text = text.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
                    text = text.replace(/^### (.*$)/gim, '<h3 style="color: #4fc3f7; margin-top: 16px; margin-bottom: 8px;">$1</h3>');
                    text = text.replace(/^## (.*$)/gim, '<h2 style="color: #4fc3f7; margin-top: 16px; margin-bottom: 8px;">$1</h2>');
                    text = text.replace(/^# (.*$)/gim, '<h1 style="color: #4fc3f7; margin-top: 16px; margin-bottom: 8px;">$1</h1>');
                    text = text.replace(/^\\* (.+)$/gim, '<li style="margin-left: 20px;">$1</li>');
                    text = text.replace(/(<li.*?<\\/li>)/s, '<ul style="margin-left: 20px; padding-left: 10px;">$1</ul>');
                    text = text.replace(/^- (.+)$/gim, '<li style="margin-left: 20px;">$1</li>');
                    text = text.replace(/(<li.*?<\\/li>)/s, '<ul style="margin-left: 20px; padding-left: 10px;">$1</ul>');
                    text = text.replace(/\\n\\n/g, '<br><br>');
                    text = text.replace(/\\n/g, '<br>');
                    text = text.replace(/`([^`]+)`/g, '<code style="background: #1a1a1a; padding: 2px 6px; border-radius: 3px; color: #4fc3f7;">$1</code>');

                    outputEl.innerHTML = text;
                }} else {{
                    outputEl.innerHTML = '<p style="color: #ff5252;">❌ Error: Invalid response format</p>';
                }}

            }} catch (error) {{
                console.error("Error:", error);
                outputEl.innerHTML = `<p style="color: #ff5252;">❌ Error: ${{error.message}}</p><p style="color: #999;">Please check API key or try again.</p>`;
            }}
        }}

        fetchGeminiResponse();
    }})();
    </script>
    """
    return html_js_content


def get_suggested_questions(df_filtered):
    """Generate context-aware suggested questions"""
    try:
        top_product = df_filtered.groupby('ProductName', observed=True)['Total Sales'].sum().idxmax()
        top_category = df_filtered.groupby('Category', observed=True)['Total Sales'].sum().idxmax()
        top_city = df_filtered.groupby('City', observed=True)['Total Sales'].sum().idxmax()
        
        # Fixed: use include_groups=False for apply
        worst_margin_prod = df_filtered.groupby('ProductName', observed=True).apply(
            lambda x: (x['Profit'].sum() / x['Total Sales'].sum() * 100) if x['Total Sales'].sum() > 0 else 0,
            include_groups=False
        ).idxmin()
        
        suggestions = [
            f"🔍 What strategies can improve sales for {worst_margin_prod}?",
            f"📈 Why is {top_product} our top performer?",
            f"💼 What growth opportunities exist in {top_category}?",
            f"🌍 How can we replicate {top_city}'s success?",
            "⚠️ What are the key risk factors?",
            "👥 Which customer segments are most valuable?",
            "📊 What seasonal patterns do you observe?",
            "🎯 What are the top 3 operational improvements?"
        ]
        return suggestions
    except:
        return [
            "📊 What are the key profitability drivers?",
            "📈 Which products show strongest growth?",
            "👥 Who are our most valuable customers?",
            "🌍 What geographical trends exist?",
            "💡 What's the biggest revenue opportunity?"
        ]


# ===== MAIN APP LOGIC =====

# Initialize session state
if 'df_master' not in st.session_state:
    st.session_state.df_master = None
if 'use_sample_data' not in st.session_state:
    st.session_state.use_sample_data = False
if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = ""


# Sidebar: Data Source Selection
st.sidebar.header("📊 Data Source Selection")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["📦 Use Preloaded Sample Dataset", "📤 Upload Your Excel File"],
    help="Start with sample data to explore all features"
)

if data_source == "📦 Use Preloaded Sample Dataset":
    st.session_state.use_sample_data = True
    sample_data = load_sample_data()
    if sample_data:
        st.session_state.df_master = process_data(sample_data["sales"], sample_data["products"], sample_data["customers"])
        if st.session_state.df_master is not None:
            st.sidebar.success("✅ Sample data loaded!")
    else:
        st.sidebar.error("Could not load sample data")

else:
    st.session_state.use_sample_data = False
    uploaded_file = st.sidebar.file_uploader("📤 Upload Excel file (.xlsx)", type=["xlsx"])
    
    if uploaded_file:
        try:
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            sheet_names = list(all_sheets.keys())

            st.sidebar.subheader("Map Your Data Sheets")
            
            def find_default_sheet(name_list, keyword):
                for i, name in enumerate(name_list):
                    if keyword.lower() in name.lower():
                        return i
                return 0 

            sales_sheet_name = st.sidebar.selectbox(
                "Sales Sheet", sheet_names, index=find_default_sheet(sheet_names, "sales")
            )
            products_sheet_name = st.sidebar.selectbox(
                "Products Sheet", sheet_names, index=find_default_sheet(sheet_names, "products")
            )
            customers_sheet_name = st.sidebar.selectbox(
                "Customers Sheet", sheet_names, index=find_default_sheet(sheet_names, "customers")
            )

            df_sales = all_sheets[sales_sheet_name]
            df_products = all_sheets[products_sheet_name]
            df_customers = all_sheets[customers_sheet_name]
            
            st.session_state.df_master = process_data(df_sales, df_products, df_customers)
            if st.session_state.df_master is not None:
                st.sidebar.success("✅ Data loaded!")

        except Exception as e:
            st.sidebar.error(f"Error: {e}")


if st.session_state.df_master is None:
    st.info("👋 Select a data source in the sidebar to begin!")
    st.stop()


df = st.session_state.df_master

# Sidebar: Filters
st.sidebar.header("🎯 Dashboard Filters")

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

start_date, end_date = st.sidebar.date_input(
    "📅 Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

all_categories = sorted(df['Category'].unique())
selected_categories = st.sidebar.multiselect(
    "📦 Categories",
    all_categories,
    default=list(all_categories)
)

all_cities = sorted(df['City'].unique())
selected_cities = st.sidebar.multiselect(
    "🌍 Cities",
    all_cities,
    default=list(all_cities)
)

df_filtered = df[
    (df['Date'].dt.date >= start_date) &
    (df['Date'].dt.date <= end_date) &
    (df['Category'].isin(selected_categories)) &
    (df['City'].isin(selected_cities))
].copy()

if df_filtered.empty:
    st.warning("⚠️ No data for selected filters. Adjust your selections.")
    st.stop()


# ===== MAIN TABS =====
tabs = st.tabs([
    "🚀 Overview & AI Summary", 
    "📈 Trend & Forecast",
    "📦 Product Analysis", 
    "👥 Customer Analysis", 
    "🌍 Geographical Analysis", 
    "💾 Data Export",
    "🤖 AI Smart Agent"
])


# ===== TAB 0: OVERVIEW & AI SUMMARY =====
with tabs[0]:
    st.subheader("📊 Key Performance Indicators")
    
    total_revenue = df_filtered['Total Sales'].sum()
    total_profit = df_filtered['Profit'].sum()
    profit_margin = (total_profit / total_revenue) * 100 if total_revenue != 0 else 0
    total_units_sold = df_filtered['Quantity'].sum()
    total_orders = df_filtered['SaleID'].nunique()
    unique_customers = df_filtered['CustomerID'].nunique()
    avg_order_value = total_revenue / total_orders if total_orders != 0 else 0
    avg_profit_per_order = total_profit / total_orders if total_orders != 0 else 0
    avg_items_per_order = total_units_sold / total_orders if total_orders != 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
    col2.metric("📈 Total Profit", f"₹{total_profit:,.0f}", delta=f"{profit_margin:.1f}%")
    col3.metric("📦 Total Units", f"{total_units_sold:,.0f}")
    col4.metric("🛒 Total Orders", f"{total_orders:,.0f}")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("👥 Customers", f"{unique_customers:,.0f}")
    col6.metric("💵 Avg Order", f"₹{avg_order_value:,.0f}")
    col7.metric("💸 Avg Profit", f"₹{avg_profit_per_order:,.0f}")
    col8.metric("📊 Items/Order", f"{avg_items_per_order:.1f}")
    
    st.markdown("---")

    col_summary, col_queries = st.columns([2, 1])
    
    with col_summary:
        st.subheader("💡 AI-Powered Executive Summary")
        st.markdown("_Gemini LLM Analysis_")
        
        data_summary = summarize_data_for_gemini(df_filtered)
        summary_prompt = f"""
        You are a senior business analyst. Analyze this data:
        
        {data_summary}
        
        Provide EXECUTIVE SUMMARY (150 words max):
        1. **Performance Overview** - one sentence summary
        2. **Top 3 KEY INSIGHTS** - specific metrics
        3. **CRITICAL RISK** - main concern
        4. **STRATEGIC RECOMMENDATION** - actionable
        
        Use bold for metrics. Values in ₹. Be data-driven.
        """
        
        gemini_summary_html = get_gemini_fetch_html(
            summary_prompt, 
            "gemini-summary-output",
            "🔄 Generating executive summary..."
        )
        components.html(gemini_summary_html, height=400, scrolling=True)
    
    with col_queries:
        st.subheader("❓ Quick Questions")
        suggested = get_suggested_questions(df_filtered)
        for i, suggestion in enumerate(suggested[:4]):
            if st.button(suggestion, key=f"quick_q_{i}", use_container_width=True):
                st.session_state.user_prompt = suggestion


# ===== TAB 1: TREND & FORECAST =====
with tabs[1]:
    st.subheader("📈 Sales & Profit Trend")
    trend_data = df_filtered.set_index('Date').resample('ME')[['Total Sales', 'Profit']].sum().reset_index()
    
    if len(trend_data) > 0:
        fig_sales_trend = px.line(
            trend_data, x='Date', y=['Total Sales', 'Profit'], 
            title="Monthly Sales and Profit Trend",
            markers=True
        )
        fig_sales_trend.update_layout(hovermode="x unified", template="plotly_dark")
        st.plotly_chart(fig_sales_trend, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔮 Advanced Sales Forecast")
    
    col_forecast_input, col_forecast_info = st.columns([1, 2])
    
    with col_forecast_input:
        forecast_months = st.slider("📅 Months to forecast:", 1, 12, 6)
    
    with col_forecast_info:
        st.info(f"📊 Data: {len(trend_data)} months | Coverage: {(len(trend_data)/12):.1f} years")
    
    if st.button("⚡ Generate Forecast", type="primary", use_container_width=True):
        if len(trend_data) < 3:
            st.warning("⚠️ Need at least 3 data points")
        else:
            try:
                with st.spinner("⏳ Calculating..."):
                    if len(trend_data) >= 12:
                        try:
                            model = ExponentialSmoothing(
                                trend_data['Total Sales'], 
                                seasonal='add', 
                                seasonal_periods=12, 
                                trend='add'
                            ).fit()
                            model_type = "Seasonal HWES"
                        except:
                            model = ExponentialSmoothing(trend_data['Total Sales'], trend='add').fit()
                            model_type = "Exponential Smoothing"
                    else:
                        model = ExponentialSmoothing(trend_data['Total Sales'], trend='add').fit()
                        model_type = "Exponential Smoothing"

                    forecast = model.forecast(forecast_months)
                    forecast_index = pd.date_range(start=trend_data['Date'].max(), periods=forecast_months + 1, freq='ME')[1:]
                    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})
                    
                    hist_df = trend_data[['Date', 'Total Sales']].copy()
                    hist_df['Type'] = 'Historical'
                    forecast_df['Type'] = 'Forecast'
                    forecast_df.rename(columns={'Forecast': 'Total Sales'}, inplace=True)
                    plot_df = pd.concat([hist_df, forecast_df], ignore_index=True)
                    
                    fig_forecast = px.line(
                        plot_df, x='Date', y='Total Sales', color='Type', 
                        title=f"Sales Forecast ({model_type})",
                        markers=True,
                        line_dash='Type'
                    )
                    fig_forecast.update_layout(hovermode="x unified", template="plotly_dark")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    col_table, col_stats = st.columns([1, 1])
                    
                    with col_table:
                        st.markdown("#### 📊 Forecast Data")
                        st.dataframe(
                            forecast_df.set_index('Date').style.format({'Total Sales': '₹{:,.0f}'}),
                            use_container_width=True
                        )
                    
                    with col_stats:
                        st.markdown("#### 📈 Statistics")
                        avg_hist = trend_data['Total Sales'].mean()
                        avg_fcst = forecast_df['Total Sales'].mean()
                        growth = ((avg_fcst - avg_hist) / avg_hist) * 100 if avg_hist > 0 else 0
                        
                        st.metric("Historical Avg", f"₹{avg_hist:,.0f}")
                        st.metric("Forecast Avg", f"₹{avg_fcst:,.0f}")
                        st.metric("Expected Growth", f"{growth:+.1f}%")
                    
                    st.markdown("---")
                    st.markdown("#### 🤖 AI Forecast Insights")
                    forecast_prompt = f"""
                    Analyze this forecast:
                    - Historical Avg: ₹{avg_hist:,.0f}/month
                    - Forecast Avg: ₹{avg_fcst:,.0f}/month
                    - Growth: {growth:+.1f}%
                    - Period: {forecast_months} months
                    
                    Provide 3 bullets on inventory/strategy implications.
                    """
                    forecast_html = get_gemini_fetch_html(
                        forecast_prompt, "gemini-forecast", "📊 Analyzing..."
                    )
                    components.html(forecast_html, height=300, scrolling=True)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ===== TAB 2: PRODUCT ANALYSIS =====
with tabs[2]:
    st.subheader("📦 Product Performance Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏆 Top 10 Products by Sales")
        sales_by_product = df_filtered.groupby('ProductName', observed=True)['Total Sales'].sum().nlargest(10).sort_values(ascending=True).reset_index()
        fig_sales_product = px.bar(
            sales_by_product, x='Total Sales', y='ProductName', orientation='h', 
            title="Top Products",
            color='Total Sales',
            color_continuous_scale='Blues'
        )
        fig_sales_product.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_sales_product, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Category Distribution")
        sales_by_category = df_filtered.groupby('Category', observed=True)['Total Sales'].sum().reset_index()
        fig_cat_pie = px.pie(
            sales_by_category, names='Category', values='Total Sales', 
            title="Sales by Category", hole=0.4
        )
        fig_cat_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_cat_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("💹 Profitability Analysis")
    
    profit_by_product = df_filtered.groupby('ProductName', observed=True).agg(
        Total_Profit=('Profit', 'sum'), 
        Total_Sales=('Total Sales', 'sum'), 
        Quantity_Sold=('Quantity', 'sum')
    ).reset_index()
    profit_by_product['Profit Margin %'] = (
        profit_by_product['Total_Profit'] / 
        profit_by_product['Total_Sales'].replace(0, np.nan)
    ) * 100
    
    fig_scatter = px.scatter(
        profit_by_product.fillna(0),
        x='Total_Sales', y='Total_Profit', text='ProductName', 
        size='Quantity_Sold',
        color='Profit Margin %', 
        color_continuous_scale=px.colors.diverging.RdYlGn,
        hover_name='ProductName', 
        title="Profit vs. Sales (Size = Quantity)",
        size_max=60
    )
    fig_scatter.update_layout(template="plotly_dark")
    fig_scatter.update_traces(textposition='top center')
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.dataframe(
        profit_by_product.sort_values('Total_Profit', ascending=False).style.format({
            'Total_Profit': '₹{:,.0f}', 'Total_Sales': '₹{:,.0f}', 'Profit Margin %': '{:.2f}%'
        }),
        use_container_width=True, height=400
    )


# ===== TAB 3: CUSTOMER ANALYSIS =====
with tabs[3]:
    st.subheader("👥 Customer Insights & RFM Analysis")
    
    st.markdown("#### 🏆 Top 15 Customers by Revenue")
    top_customers = df_filtered.groupby(['CustomerName', 'City'], observed=True).agg({
        'Total Sales': 'sum',
        'SaleID': 'count',
        'Profit': 'sum'
    }).rename(columns={'SaleID': 'Order Count'}).nlargest(15, 'Total Sales').reset_index()
    
    top_customers['Profit Margin %'] = (
        top_customers['Profit'] / top_customers['Total Sales'].replace(0, np.nan)
    ) * 100
    
    st.dataframe(
        top_customers.style.format({
            'Total Sales': '₹{:,.0f}', 
            'Profit': '₹{:,.0f}',
            'Profit Margin %': '{:.2f}%'
        }),
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("#### 💳 RFM Customer Segmentation")
    
    # RFM Analysis - FIXED (no observed in pd.cut())
    reference_date = df_filtered['Date'].max() + timedelta(days=1)
    
    rfm = df_filtered.groupby('CustomerID', observed=True).agg({
        'Date': lambda x: (reference_date - x.max()).days,
        'SaleID': 'count',
        'Total Sales': 'sum'
    }).rename(columns={
        'Date': 'Recency',
        'SaleID': 'Frequency',
        'Total Sales': 'Monetary'
    }).reset_index()
    
    rfm['Score'] = rfm['Recency'].rank() + rfm['Frequency'].rank() + rfm['Monetary'].rank()
    
    # FIXED: No 'observed' parameter in pd.cut()
    rfm['Segment'] = pd.cut(
        rfm['Score'], 
        bins=3, 
        labels=['Low Value', 'Medium Value', 'High Value']
    )
    
    rfm_summary = rfm['Segment'].value_counts()
    
    fig_segment = px.pie(
        values=rfm_summary.values, 
        names=rfm_summary.index, 
        title="Customer Value Segmentation",
        hole=0.3
    )
    fig_segment.update_layout(template="plotly_dark")
    st.plotly_chart(fig_segment, use_container_width=True)
    
    st.markdown("#### 📊 RFM Details")
    rfm_display = rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']].sort_values('Monetary', ascending=False).head(20)
    st.dataframe(
        rfm_display.style.format({
            'Recency': '{:.0f} days',
            'Frequency': '{:.0f} orders',
            'Monetary': '₹{:,.0f}'
        }),
        use_container_width=True
    )


# ===== TAB 4: GEOGRAPHICAL ANALYSIS =====
with tabs[4]:
    st.subheader("🌍 Geographical Sales Analysis")
    
    sales_by_city = df_filtered.groupby('City', observed=True).agg({
        'Total Sales': 'sum',
        'Profit': 'sum',
        'SaleID': 'count'
    }).nlargest(15, 'Total Sales').reset_index()
    
    sales_by_city['Profit Margin %'] = (
        sales_by_city['Profit'] / sales_by_city['Total Sales'].replace(0, np.nan)
    ) * 100
    
    fig_city_sales = px.bar(
        sales_by_city.sort_values('Total Sales', ascending=True), 
        x='Total Sales', y='City', orientation='h', 
        title="Top 15 Cities by Sales",
        color='Profit Margin %',
        color_continuous_scale='Viridis'
    )
    fig_city_sales.update_layout(template="plotly_dark")
    st.plotly_chart(fig_city_sales, use_container_width=True)
    
    st.markdown("---")
    st.dataframe(
        sales_by_city.sort_values('Total Sales', ascending=False).style.format({
            'Total Sales': '₹{:,.0f}',
            'Profit': '₹{:,.0f}',
            'Profit Margin %': '{:.2f}%'
        }),
        use_container_width=True
    )


# ===== TAB 5: DATA EXPORT =====
with tabs[5]:
    st.subheader("📊 Custom Pivot Tables & Data Export")
    
    st.markdown("#### 🔧 Build Custom Pivot Table")
    all_cols = [col for col in df_filtered.columns if col != 'SaleID']
    numeric_cols = df_filtered.select_dtypes(include=['number']).columns.tolist()
    default_values = 'Total Sales' if 'Total Sales' in numeric_cols else numeric_cols[0]
    
    col1, col2 = st.columns(2)
    with col1:
        pivot_rows = st.multiselect("Rows", options=all_cols, default=['Category'])
    with col2:
        pivot_cols = st.multiselect("Columns", options=all_cols, default=['Month-Year'])
    col3, col4 = st.columns(2)
    with col3:
        pivot_val = st.selectbox("Value", options=numeric_cols, index=numeric_cols.index(default_values))
    with col4:
        agg_func = st.selectbox("Aggregation", options=['sum', 'mean', 'count', 'median', 'max', 'min'])
        
    if pivot_rows and pivot_val:
        try:
            pivot_df = pd.pivot_table(
                df_filtered, 
                index=pivot_rows, 
                columns=pivot_cols if pivot_cols else None,
                values=pivot_val, 
                aggfunc=agg_func, 
                fill_value=0,
                observed=True
            )
            st.dataframe(pivot_df, use_container_width=True)
            
            st.download_button(
                label="📥 Download Pivot as CSV", 
                data=convert_df_to_csv(pivot_df.reset_index()),
                file_name="pivot_table.csv", 
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("👉 Select Rows and Value to create pivot")

    st.markdown("---")
    
    with st.expander("📂 Export Raw Data", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Filtered Data CSV", 
                data=convert_df_to_csv(df_filtered),
                file_name="filtered_data.csv", 
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="📥 All Data CSV", 
                data=convert_df_to_csv(df),
                file_name="all_data.csv", 
                mime="text/csv"
            )
        
        st.dataframe(df_filtered, use_container_width=True, height=400)


# ===== TAB 6: AI SMART AGENT =====
with tabs[6]:
    st.subheader("🤖 AI Smart Agent")
    st.markdown("Ask questions about your data. AI uses current filtered data to answer.")
    
    st.markdown("#### 💡 Suggested Questions:")
    
    suggested_qs = get_suggested_questions(df_filtered)
    cols = st.columns(2)
    for i, question in enumerate(suggested_qs):
        clean_q = question.replace("🔍 ", "").replace("📈 ", "").replace("💼 ", "").replace("🌍 ", "").replace("⚠️ ", "").replace("👥 ", "").replace("📊 ", "").replace("🎯 ", "")
        with cols[i % 2]:
            if st.button(f"❓ {clean_q[:50]}...", key=f"q_{i}", use_container_width=True):
                st.session_state.user_prompt = clean_q
    
    st.markdown("---")
    
    user_prompt = st.text_area(
        "Your Question:",
        value=st.session_state.get('user_prompt', ''),
        placeholder="Ask anything about products, customers, trends...",
        height=100
    )
    
    col_ask, col_clear = st.columns([4, 1])
    
    with col_ask:
        if st.button("🚀 Get Answer", type="primary", use_container_width=True):
            if not user_prompt.strip():
                st.warning("⚠️ Enter a question")
            else:
                st.session_state.user_prompt = ""
                data_summary = summarize_data_for_gemini(df_filtered)
                qa_prompt = f"""
                You are a senior business analyst answering a data question.
                
                DATA:
                {data_summary}
                
                QUESTION:
                "{user_prompt}"
                
                Answer comprehensively:
                - Start with key insight
                - Use specific numbers
                - Provide 2-3 recommendations
                - Identify opportunities/risks
                
                Format: Markdown. Values: ₹. Be concise & data-driven.
                """
                
                gemini_qa_html = get_gemini_fetch_html(
                    qa_prompt,
                    "gemini-qa-output",
                    "🔄 Smart Agent analyzing..."
                )
                
                st.markdown("### 🤖 Response:")
                components.html(gemini_qa_html, height=700, scrolling=True)
                
                st.markdown("---")
                st.markdown("**Data Context:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Records", f"{len(df_filtered):,}")
                col2.metric("Period", f"{df_filtered['Date'].min().date()} - {df_filtered['Date'].max().date()}")
                col3.metric("Categories", len(selected_categories))
    
    with col_clear:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.user_prompt = ""


# ===== FOOTER =====
st.markdown("---")
st.caption(
    "🚀 AI-Powered Sales Dashboard | "
    "🤖 Gemini 2.0 Flash | "
    "📊 Analytics & Forecasting | "
    "💼 Data-Driven Decisions | "
    "© 2025"
)
