import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analysis_functions_updated import (
    calculate_wow_yoy, perform_hierarchical_analysis_updated, rank_combinations,
    detect_significant_changes, detect_masked_issues_improved, calculate_impact, prioritize_findings,
    perform_multi_dimensional_breakdown_advanced, perform_cross_metric_impact_analysis_advanced,
    detect_hidden_issues_advanced, get_business_context
)
import os

# Optional import for Google Generative AI
try:
    import google.generativeai as genai
    from llm_integration import generate_business_insights_with_llm, generate_enhanced_root_cause_analysis, create_analysis_summary, create_dataframe_summary
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# Set page configuration
st.set_page_config(page_title="Data Science Assessment Tool", layout="wide")

# Title
st.title("📊 Data Science Assessment Tool")
st.markdown("A **dataset-agnostic** analysis tool for detecting performance changes and identifying business drivers. Think of it as your data detective! 🕵️‍♀️")

# LLM Configuration in sidebar
st.sidebar.header("🤖 AI Enhancement")

if not GEMINI_AVAILABLE:
    st.sidebar.error("⚠️ Google Generative AI package not installed. Install with: `pip install google-generativeai`")
    use_llm = False
else:
    use_llm = st.sidebar.toggle("Enable AI-Powered Insights", value=False, help="Use Google Gemini 2.5 Pro for enhanced business insights and root cause analysis")

    if use_llm:
        # Check for API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            st.sidebar.error("⚠️ GEMINI_API_KEY environment variable not found. Please set it to use AI features.")
            use_llm = False
        else:
            try:
                genai.configure(api_key=gemini_api_key)
                st.sidebar.success("✅ AI features enabled")
            except Exception as e:
                st.sidebar.error(f"❌ Error configuring Gemini API: {str(e)}")
                use_llm = False

# Sidebar for file upload and configuration
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

# Initialize session state for analysis results
if 'analysis_completed' not in st.session_state:
    st.session_state.analysis_completed = False
if 'hierarchical_results' not in st.session_state:
    st.session_state.hierarchical_results = pd.DataFrame()
if 'impact_results' not in st.session_state:
    st.session_state.impact_results = pd.DataFrame()
if 'full_df_for_visualizations' not in st.session_state:
    st.session_state.full_df_for_visualizations = pd.DataFrame()
if 'metrics' not in st.session_state:
    st.session_state.metrics = []
if 'dimensions' not in st.session_state:
    st.session_state.dimensions = []
if 'date_column' not in st.session_state:
    st.session_state.date_column = None

# Check if file is uploaded
if uploaded_file is None:
    # No file uploaded - show empty state
    st.markdown("---")
    st.markdown("### 📁 No Dataset Uploaded")
    st.info("👆 Please upload a CSV file using the file uploader in the sidebar to begin your analysis.")
    st.markdown("**What you can do:**")
    st.markdown("- Upload a CSV file with your business data")
    st.markdown("- The tool will automatically detect metrics (numbers) and dimensions (categories)")
    st.markdown("- Configure analysis parameters in the sidebar")
    st.markdown("- Run comprehensive analysis to find insights in your data")
    
    # Clear session state when no file is uploaded
    st.session_state.analysis_completed = False
    st.session_state.hierarchical_results = pd.DataFrame()
    st.session_state.impact_results = pd.DataFrame()
    st.session_state.full_df_for_visualizations = pd.DataFrame()
    st.session_state.metrics = []
    st.session_state.dimensions = []
    st.session_state.date_column = None
    
    st.stop()

# File is uploaded - proceed with analysis
df = pd.read_csv(uploaded_file)

# Display basic info about the dataset
st.header("📋 Dataset Overview")
st.write(f"Your dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns**.")
st.write("Here's a sneak peek at your data:")
st.dataframe(df.head())

# Automatically identify metrics and dimensions
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Try to identify date column
date_column = None
for col in df.columns:
    try:
        pd.to_datetime(df[col])
        date_column = col
        break
    except:
        continue

if date_column:
    df[date_column] = pd.to_datetime(df[date_column])
    categorical_columns = [col for col in categorical_columns if col != date_column]

st.sidebar.header("🔍 Column Identification")
st.sidebar.write(f"**Date column:** `{date_column}`")
st.sidebar.write(f"**Metrics (numbers):** `{', '.join(numeric_columns)}`")
st.sidebar.write(f"**Dimensions (categories):** `{', '.join(categorical_columns)}`")

# Allow user to modify the identification
metrics = st.sidebar.multiselect("Which numbers do you want to analyze?", numeric_columns, default=numeric_columns)
dimensions = st.sidebar.multiselect("Which categories do you want to break down by?", categorical_columns, default=categorical_columns[:4] if len(categorical_columns) >= 4 else categorical_columns)

# Store in session state
st.session_state.metrics = metrics
st.session_state.dimensions = dimensions
st.session_state.date_column = date_column

# Configuration parameters
st.sidebar.header("⚙️ Analysis Parameters")
threshold = st.sidebar.slider("How big of a change is 'significant' (in %)?", 1, 50, 5)
alpha = st.sidebar.slider("Statistical Significance Level (alpha)", 0.01, 0.10, 0.05, 0.01)
top_n = st.sidebar.slider("Show top/bottom how many results?", 3, 10, 5)

# Business criticality weights
st.sidebar.header("💰 Business Importance Weights")
st.sidebar.write("Tell me how important each number is for your business (higher = more important).")
business_weights = {}
for metric in metrics:
    business_weights[metric] = st.sidebar.slider(f"Importance of **{metric}**", 0.1, 5.0, 1.0)

# Main analysis
if st.button("🚀 Run Analysis!", type="primary"):
    if date_column is None:
        st.error("Oops! I can't find a 'Date' column. Please make sure your dataset has one so I can track changes over time.")
    elif len(metrics) == 0 or len(dimensions) == 0:
        st.error("Please select at least one 'number to analyze' (metric) and one 'category to break down by' (dimension) to start the analysis.")
    else:
        with st.spinner("🕵️‍♀️ Analyzing your data... This might take a moment for big datasets!"):
            try:
                # Perform hierarchical analysis
                hierarchical_results = perform_hierarchical_analysis_updated(df, dimensions, metrics, date_column, alpha)
                
                # Store results in session state
                st.session_state.hierarchical_results = hierarchical_results
                st.session_state.full_df_for_visualizations = df.copy()
                st.session_state.analysis_completed = True
                st.session_state.threshold = threshold
                st.session_state.alpha = alpha
                st.session_state.top_n = top_n
                st.session_state.business_weights = business_weights
                
                if hierarchical_results.empty:
                    st.error("No analysis results generated. Please check your data and try again. Make sure your dataset has enough historical data for WoW/YoY calculations.")
                    
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {str(e)}")
                st.write("This might be due to data format issues or insufficient data for certain calculations. Please check your data and try again.")

# Display analysis results (moved outside button click to preserve them)
if st.session_state.analysis_completed and not st.session_state.hierarchical_results.empty:
    # Get stored parameters
    threshold = st.session_state.get('threshold', 5)
    alpha = st.session_state.get('alpha', 0.05)
    top_n = st.session_state.get('top_n', 5)
    business_weights = st.session_state.get('business_weights', {})
    hierarchical_results = st.session_state.hierarchical_results
    
    st.header("📈 Hierarchical Analysis: What's Changing?")
    st.markdown("I'm looking at how your numbers (metrics) are changing over time, both week-to-week and year-to-year, across different categories (dimensions).")
    
    st.subheader("📊 Week-over-Week (WoW) and Year-over-Year (YoY) Changes by Level")
    st.markdown("This table shows changes at different levels: Overall (everything combined), Level 1 (single categories), and Level 2 (category combinations).")
    
    # Show a sample of results with better formatting
    display_df = hierarchical_results.head(20)  # Show first 20 results
    st.dataframe(display_df.style.format({
        "Latest_WoW_Change": "{:.2f}%", 
        "Latest_YoY_Change": "{:.2f}%",
        "Latest_Value": "{:,.2f}",
        "P_Value": "{:.3f}"
    }))
    
    if len(hierarchical_results) > 20:
        st.info(f"Showing first 20 results out of {len(hierarchical_results)} total results.")
    
    # Detect significant changes
    st.header("🚨 Significant Change Detection: Big Shifts!")
    st.markdown(f"I'm flagging changes that are bigger than **{threshold}%** or are statistically unusual (with a significance level of **{alpha*100:.0f}%**).")
    significant_results = hierarchical_results.copy()
    significant_results = detect_significant_changes(significant_results, "Metric", "Latest_WoW_Change", threshold, alpha)
    
    significant_changes = significant_results[significant_results["Significant_Change_Flag"] == True]
    statistically_significant_changes = significant_results[significant_results["Statistical_Significance_Flag"] == True]

    if not significant_changes.empty:
        st.subheader(f"🔥 These are the big WoW changes I found (more than {threshold}%):")
        st.dataframe(significant_changes[["Level", "Dimension_Combination", "Metric", "Latest_WoW_Change", "Latest_Value", "Is_Statistically_Significant", "P_Value"]].style.format({"Latest_WoW_Change": "{:.2f}%", "Latest_Value": "{:,.2f}", "P_Value": "{:.3f}"}))
        
        # Add explanations for significant changes
        st.markdown("**What this means:**")
        for _, row in significant_changes.head(3).iterrows():
            change_direction = "increased" if row["Latest_WoW_Change"] > 0 else "decreased"
            sig_text = "and is statistically significant!" if row["Is_Statistically_Significant"] else "but is NOT statistically significant."
            st.write(f"• **{row['Metric']}** for **{row['Dimension_Combination']}** {change_direction} by **{abs(row['Latest_WoW_Change']):.1f}%** this week! {sig_text}")
    else:
        st.info(f"Good news! No really big week-over-week changes detected (above {threshold}%). Your business seems stable!")
    
    if not statistically_significant_changes.empty:
        st.subheader(f"✨ These changes are statistically significant (p < {alpha:.2f}):")
        st.dataframe(statistically_significant_changes[["Level", "Dimension_Combination", "Metric", "Latest_WoW_Change", "Latest_Value", "Is_Statistically_Significant", "P_Value"]].style.format({"Latest_WoW_Change": "{:.2f}%", "Latest_Value": "{:,.2f}", "P_Value": "{:.3f}"}))
        st.markdown("**What this means:** These changes are unlikely to be due to random chance. They are real shifts in your data!")
    else:
        st.info(f"No statistically significant changes detected (p < {alpha:.2f}).")

    # Ranking top/bottom performers
    st.header("🏆 Top/Bottom Performers: Who's Winning/Losing?")
    st.markdown(f"Here are the top and bottom **{top_n}** performers based on their week-over-week changes.")
    for metric in st.session_state.metrics:
        metric_data = hierarchical_results[hierarchical_results["Metric"] == metric]
        if not metric_data.empty:
            st.subheader(f"🎯 For **{metric}**:")
            top_performers, bottom_performers = rank_combinations(metric_data, "Metric", "Latest_WoW_Change", top_n)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**🚀 Top Performers (WoW Increase)**")
                if not top_performers.empty:
                    st.dataframe(top_performers[["Level", "Dimension_Combination", "Latest_WoW_Change"]].style.format({"Latest_WoW_Change": "{:.2f}%"}))
                    # Add explanation
                    best_performer = top_performers.iloc[0]
                    st.success(f"🌟 Best: **{best_performer['Dimension_Combination']}** is up **{best_performer['Latest_WoW_Change']:.1f}%**!")
                else:
                    st.info("No top performers found for this metric.")
            with col2:
                st.write("**📉 Bottom Performers (WoW Decrease)**")
                if not bottom_performers.empty:
                    st.dataframe(bottom_performers[["Level", "Dimension_Combination", "Latest_WoW_Change"]].style.format({"Latest_WoW_Change": "{:.2f}%"}))
                    # Add explanation
                    worst_performer = bottom_performers.iloc[0]
                    st.warning(f"⚠️ Needs attention: **{worst_performer['Dimension_Combination']}** is down **{abs(worst_performer['Latest_WoW_Change']):.1f}%**!")
                else:
                    st.info("No bottom performers found for this metric.")

    # Masked issue detection
    st.header("🎭 Masked Issue Detection: Hidden Surprises!")
    st.markdown("Sometimes, the overall numbers look fine, but hidden problems (or successes!) are lurking underneath. I'm trying to find those for you.")
    masked_issues = detect_masked_issues_improved(st.session_state.full_df_for_visualizations, st.session_state.dimensions, st.session_state.metrics, st.session_state.date_column)
    if masked_issues:
        st.subheader("🔍 I found some hidden issues!")
        for issue in masked_issues:
            st.warning(f"🎭 **{issue['Issue']}**")
            with st.expander("Tell me more!"):
                st.write(f"- **Metric:** {issue['Metric']}")
                st.write(f"- **Category:** {issue['Dimension']}")
                st.write(f"- **Offsetting Groups:** {issue['Positive_Segment']} (+{issue['Positive_Change']:.1f}%) vs {issue['Negative_Segment']} ({issue['Negative_Change']:.1f}%)")
                st.write(f"- **Overall Change:** {issue['Overall_Change']:.2f}%")
                st.markdown("**Why this matters:** When overall numbers look stable, you might miss important trends happening in specific segments of your business!")
    else:
        st.info("Great! No masked issues detected. Your overall numbers truly reflect what's happening in all segments.")
    
    # Impact-based prioritization
    st.header("💥 Impact-Based Prioritization: What Matters Most?")
    st.markdown("From all the changes, I'm showing you the ones that have the biggest impact on your business, based on how important you said each number is.")
    impact_results = calculate_impact(significant_results, "Metric", "Latest_WoW_Change", "Latest_Value", business_weights)
    
    # Store impact results in session state
    st.session_state.impact_results = impact_results
    
    # Filter for key metrics if they exist
    key_metrics_for_prioritization = [m for m in ["shoppers", "revenue", "profit"] if m in st.session_state.metrics]
    if key_metrics_for_prioritization:
        prioritized_results = prioritize_findings(impact_results, key_metrics_for_prioritization, top_n)
        if not prioritized_results.empty:
            st.subheader(f"🎯 Top {top_n} Most Impactful Findings:")
            display_cols = ["Level", "Dimension_Combination", "Metric", "Latest_WoW_Change", "Latest_Value", "Impact"]
            # FIXED: Ensure exactly top_n results are shown
            prioritized_display = prioritized_results[display_cols].sort_values("Impact", ascending=False).head(top_n)
            st.dataframe(prioritized_display.style.format({"Latest_WoW_Change": "{:.2f}%", "Latest_Value": "{:,.2f}", "Impact": "{:.2f}"}))
            
            # Add explanation for top impact
            top_impact = prioritized_display.iloc[0]
            st.info(f"💡 **Biggest Impact:** {top_impact['Dimension_Combination']} in {top_impact['Metric']} (Impact Score: {top_impact['Impact']:.1f})")
        else:
            st.info("No impactful findings available for key metrics (shoppers, revenue, profit). Try adjusting your weights or threshold.")
    else:
        st.info("Key metrics (shoppers, revenue, profit) not found in your selected metrics. Showing top impactful findings across all selected metrics.")
        prioritized_results = impact_results.sort_values("Impact", ascending=False).head(top_n)
        if not prioritized_results.empty:
            display_cols = ["Level", "Dimension_Combination", "Metric", "Latest_WoW_Change", "Latest_Value", "Impact"]
            st.dataframe(prioritized_results[display_cols].style.format({"Latest_WoW_Change": "{:.2f}%", "Latest_Value": "{:,.2f}", "Impact": "{:.2f}"}))
        else:
            st.info("No impactful findings across all selected metrics. Try adjusting your weights or threshold.")

# Dynamic Visualizations Section (Outside the main analysis button)
if st.session_state.analysis_completed and not st.session_state.hierarchical_results.empty:
    st.header("📊 Visualizations: See the Story!")
    
    # WoW Change distribution
    st.subheader("📈 How Big Are the Week-over-Week Changes?")
    st.markdown("This graph shows how often different sizes of week-over-week changes happen across all your data.")
    
    # Allow user to select metric for histogram - FIXED with unique key
    selected_hist_metric = st.selectbox("Select metric for WoW Change Distribution:", 
                                       st.session_state.metrics, 
                                       key="histogram_metric_selector")
    
    if selected_hist_metric:
        hist_data = st.session_state.hierarchical_results[st.session_state.hierarchical_results["Metric"] == selected_hist_metric]["Latest_WoW_Change"].dropna()
        hist_data = hist_data[hist_data != 0]  # Remove zero changes for better visualization
        
        if not hist_data.empty and len(hist_data) > 1:
            fig = px.histogram(hist_data, title=f"Distribution of Week-over-Week Changes for {selected_hist_metric}", 
                             nbins=min(30, len(hist_data.unique())),
                             labels={
                                 "value": "WoW Change (%)",
                                 "count": "Number of Occurrences"
                             },
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(xaxis_title="WoW Change (%)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True, key=f"histogram_{selected_hist_metric}")
            
            # Add interpretation
            avg_change = hist_data.mean()
            max_change = hist_data.max()
            min_change = hist_data.min()
            st.markdown(f"**What this shows:** For **{selected_hist_metric}**, the average change is **{avg_change:.1f}%**, with the biggest increase at **{max_change:.1f}%** and biggest decrease at **{min_change:.1f}%**.")
        else:
            st.info(f"Not enough varied data for {selected_hist_metric} to show WoW Change Distribution. This might happen with very stable data.")
    
    # Impact vs Change scatter plot
    if not st.session_state.impact_results.empty and "Impact" in st.session_state.impact_results.columns:
        st.subheader("💥 Impact vs WoW Change: What's Driving Value?")
        plot_data = st.session_state.impact_results[st.session_state.impact_results["Impact"] > 0] # Only show positive impact for clarity
        
        if not plot_data.empty and len(plot_data) > 1:
            fig = px.scatter(plot_data, x="Latest_WoW_Change", y="Impact", 
                           color="Metric", hover_data=["Level", "Dimension_Combination", "Latest_Value"],
                           title="Business Impact vs. Week-over-Week Change",
                           labels={
                               "Latest_WoW_Change": "WoW Change (%)",
                               "Impact": "Business Impact Score"
                           },
                           size="Impact", # Size of marker by impact
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(xaxis_title="WoW Change (%)", yaxis_title="Business Impact Score")
            st.plotly_chart(fig, use_container_width=True, key="impact_scatter")
            
            # Add interpretation
            high_impact_items = plot_data[plot_data["Impact"] > plot_data["Impact"].quantile(0.8)]
            if not high_impact_items.empty:
                st.markdown("**What this shows:** Each dot represents a metric-dimension combination. Bigger dots = higher business impact. Look for dots far from the center - they represent the most significant changes!")
        else:
            st.info("Not enough data with positive impact to display in scatter plot. Try adjusting your weights or threshold.")
    else:
        st.info("Impact data not available for scatter plot.")
    
    # Time series for selected metrics (Optimized to prevent full reload)
    st.subheader("⏳ Time Series Analysis: How Numbers Change Over Time")
    st.markdown("Pick a number (metric) to see how it has changed day by day.")
    
    # FIXED with unique key
    selected_metric_ts = st.selectbox("Which number do you want to see over time?", 
                                     st.session_state.metrics, 
                                     key="timeseries_metric_selector")
    
    if selected_metric_ts and not st.session_state.full_df_for_visualizations.empty:
        # Use the stored dataframe for time series plotting
        time_series_data = st.session_state.full_df_for_visualizations.groupby(st.session_state.date_column)[selected_metric_ts].sum().reset_index()
        fig = px.line(time_series_data, x=st.session_state.date_column, y=selected_metric_ts, 
                     title=f"Daily {selected_metric_ts} Over Time",
                     labels={
                         st.session_state.date_column: "Date",
                         selected_metric_ts: f"{selected_metric_ts} Value"
                     },
                     color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(xaxis_title="Date", yaxis_title=f"{selected_metric_ts} Value")
        st.plotly_chart(fig, use_container_width=True, key=f"timeseries_{selected_metric_ts}")
        
        # Add trend interpretation
        # Ensure enough data points for trend calculation
        if len(time_series_data) >= 60: # At least 2 months of data for a meaningful trend
            recent_avg = time_series_data[selected_metric_ts].tail(30).mean()
            older_avg = time_series_data[selected_metric_ts].head(30).mean()
            trend_change = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
            
            if trend_change > 5:
                st.success(f"📈 **Trend:** {selected_metric_ts} is trending upward! Recent 30-day average is {trend_change:.1f}% higher than the beginning.")
            elif trend_change < -5:
                st.warning(f"📉 **Trend:** {selected_metric_ts} is trending downward. Recent 30-day average is {abs(trend_change):.1f}% lower than the beginning.")
            else:
                st.info(f"📊 **Trend:** {selected_metric_ts} is relatively stable with {trend_change:.1f}% change from beginning to recent period.")
        else:
            st.info("Not enough data to reliably calculate a long-term trend for this metric.")

# Additional information
st.sidebar.header("💡 About This Tool")
st.sidebar.info(
    "This tool helps you understand your business data better by automatically finding important changes, "
    "hidden issues, and what's driving your numbers up or down. It's like having a data detective at your fingertips!"
)

st.sidebar.header("🎯 How to Interpret Results")
st.sidebar.markdown("""
**📈 WoW Change:** Week-over-week percentage change
**📅 YoY Change:** Year-over-year percentage change  
**🏆 Top Performers:** Segments with biggest increases
**📉 Bottom Performers:** Segments with biggest decreases
**🎭 Masked Issues:** Hidden problems where overall looks stable but segments are changing
**💥 Impact Score:** How much a change affects your business (bigger = more important)
**✨ Statistical Significance:** Whether a change is likely real or just random chance (p-value < alpha means it's significant!)
""")



# Advanced Analysis Section
if st.session_state.analysis_completed:
    st.sidebar.header("🔬 Advanced Analysis")
    st.sidebar.markdown("Select a specific change to deep dive into its root causes.")

    # Prepare data for selection
    if not st.session_state.hierarchical_results.empty:
        # Create a unique identifier for each change for the dropdown
        def create_change_id(row):
            return f"{row['Metric']} for {row['Dimension_Combination']} ({row['Latest_WoW_Change']:.2f}%) - {row['Level']}"
        
        st.session_state.hierarchical_results["Change_ID"] = st.session_state.hierarchical_results.apply(create_change_id, axis=1)
        
        selected_change_id = st.sidebar.selectbox(
            "Choose a change to analyze:",
            options=["Select a change..."] + st.session_state.hierarchical_results["Change_ID"].tolist(),
            key="advanced_analysis_selector"
        )

    else:
        st.sidebar.info("Run the initial analysis first to enable advanced root cause analysis.")

# Display Advanced Analysis content based on selection (moved outside sidebar)
if st.session_state.analysis_completed and not st.session_state.hierarchical_results.empty:
    # Check if a change is selected for advanced analysis
    if 'advanced_analysis_selector' in st.session_state and st.session_state.advanced_analysis_selector != "Select a change...":
        selected_change_id = st.session_state.advanced_analysis_selector
        selected_change_row = st.session_state.hierarchical_results[
            st.session_state.hierarchical_results["Change_ID"] == selected_change_id
        ].iloc[0]

        st.header("🔬 Root Cause Analysis: The Full Story!")
        st.markdown("Let's dig deeper into why this change happened and what it means for your business.")

        # Multi-Dimensional Breakdown Section
        st.subheader("🔍 Multi-Dimensional Breakdown")
        st.markdown("This shows how other related metrics changed within the same segment.")
        
        try:
            with st.spinner("Analyzing multi-dimensional breakdown..."):
                multi_dim_narrative = perform_multi_dimensional_breakdown_advanced(
                    st.session_state.full_df_for_visualizations,
                    selected_change_row,
                    st.session_state.date_column,
                    st.session_state.metrics,
                    st.session_state.dimensions,
                    alpha
                )
                for line in multi_dim_narrative:
                    st.markdown(line)
        except Exception as e:
            st.error(f"Error in multi-dimensional breakdown: {str(e)}")

        # Cross-Metric Impact Analysis Section
        st.subheader("🔗 Cross-Metric Impact Analysis")
        st.markdown("Understanding the upstream and downstream factors that influenced this change.")
        
        try:
            with st.spinner("Analyzing cross-metric impacts..."):
                cross_metric_narrative = perform_cross_metric_impact_analysis_advanced(
                    st.session_state.full_df_for_visualizations,
                    selected_change_row,
                    st.session_state.date_column,
                    st.session_state.metrics,
                    st.session_state.dimensions,
                    alpha
                )
                for line in cross_metric_narrative:
                    st.markdown(line)
        except Exception as e:
            st.error(f"Error in cross-metric impact analysis: {str(e)}")

        # Business Context and Recommendations Section
        if use_llm:
            st.subheader("💡 Business Context and Recommendations (Powered by AI)")
        else:
            st.subheader("💡 Business Context and Recommendations")
        
        if use_llm:
            try:
                with st.spinner("🤖 AI is analyzing all data to generate comprehensive business insights..."):
                    # Create comprehensive analysis summary for LLM
                    analysis_summary = create_analysis_summary(st.session_state.hierarchical_results, st.session_state.metrics, st.session_state.dimensions)
                    impact_results = st.session_state.get('impact_results', pd.DataFrame())
                    masked_issues = detect_masked_issues_improved(st.session_state.full_df_for_visualizations, st.session_state.dimensions, st.session_state.metrics, st.session_state.date_column)
                    
                    # Generate LLM insights
                    llm_insights = generate_business_insights_with_llm(analysis_summary, st.session_state.hierarchical_results, impact_results, masked_issues)
                    st.markdown(llm_insights)
                    
                # AI-Enhanced Deep Dive Analysis (moved under Business Context)
                st.markdown("---")
                st.markdown("### 🧠 AI-Enhanced Deep Dive Analysis")
                
                try:
                    with st.spinner("🤖 AI is conducting deep root cause analysis..."):
                        # Generate enhanced root cause analysis with forecasting
                        enhanced_analysis = generate_enhanced_root_cause_analysis(
                            selected_change_row, 
                            multi_dim_narrative, 
                            cross_metric_narrative, 
                            st.session_state.hierarchical_results,
                            create_dataframe_summary(st.session_state.full_df_for_visualizations),
                            st.session_state.full_df_for_visualizations,  # Pass full dataframe for forecasting
                            st.session_state.date_column  # Pass date column for forecasting
                        )
                        
                        st.markdown(enhanced_analysis)
                        
                except Exception as e:
                    st.error(f"Error generating enhanced root cause analysis: {str(e)}")
                    st.markdown("The standard analysis sections above provide the available insights.")
                    
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
                st.markdown("Falling back to standard business context analysis...")
                # Fallback to original logic
                change_direction = "increase" if selected_change_row["Latest_WoW_Change"] > 0 else "decline"
                business_context = get_business_context(selected_change_row["Metric"].lower(), change_direction)
                
                if business_context["flag"] == "attention":
                    st.error("🚨 **Requires Immediate Attention!**")
                elif business_context["flag"] == "positive":
                    st.success("✅ **Positive Trend to Amplify!**")
                else:
                    st.info("ℹ️ **Neutral Trend - Further Investigation Recommended.**")

                st.markdown("**Possible Business Reasons:**")
                for reason in business_context["reasons"]:
                    st.markdown(f"- {reason}")

                st.markdown("**Actionable Recommendations:**")
                for rec in business_context["recommendations"]:
                    st.markdown(f"- {rec}")
        else:
            st.markdown("Here are some possible business reasons and actionable recommendations for this change.")
            
            try:
                change_direction = "increase" if selected_change_row["Latest_WoW_Change"] > 0 else "decline"
                business_context = get_business_context(selected_change_row["Metric"].lower(), change_direction)
                
                if business_context["flag"] == "attention":
                    st.error("🚨 **Requires Immediate Attention!**")
                elif business_context["flag"] == "positive":
                    st.success("✅ **Positive Trend to Amplify!**")
                else:
                    st.info("ℹ️ **Neutral Trend - Further Investigation Recommended.**")

                st.markdown("**Possible Business Reasons:**")
                for reason in business_context["reasons"]:
                    st.markdown(f"- {reason}")

                st.markdown("**Actionable Recommendations:**")
                for rec in business_context["recommendations"]:
                    st.markdown(f"- {rec}")
            except Exception as e:
                st.error(f"Error in business context analysis: {str(e)}")
                business_context = get_business_context(selected_change_row["Metric"].lower(), change_direction)
                
                if business_context["flag"] == "attention":
                    st.error("🚨 **Requires Immediate Attention!**")
                elif business_context["flag"] == "positive":
                    st.success("✅ **Positive Trend to Amplify!**")
                else:
                    st.info("ℹ️ **Neutral Trend - Further Investigation Recommended.**")

                st.markdown("**Possible Business Reasons:**")
                for reason in business_context["reasons"]:
                    st.markdown(f"- {reason}")

                st.markdown("**Actionable Recommendations:**")
                for rec in business_context["recommendations"]:
                    st.markdown(f"- {rec}")
            except Exception as e:
                st.error(f"Error in business context analysis: {str(e)}")




