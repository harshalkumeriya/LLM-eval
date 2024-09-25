import streamlit as st
import random
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        font-family: Arial, sans-serif;
        color: white;
    }

    .metric-tile {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 150px;
        height: 150px;
        border: 1px solid #ccc;
        padding: 10px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2); 
        margin: 10px;  /* Add margin to create space between tiles */
    }

    .metric-name {
        font-weight: bold;
        font-size: 1.5em; /* Bigger font size */
        color: black;
    }

    .metric-value {
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Upload and Configure")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])
framework = st.sidebar.radio("Select Framework", ("Framework A", "Framework B", "Framework C"))
metrics = st.sidebar.multiselect("Select Metrics", ("Metric 1", "Metric 2", "Metric 3", "Metric 4", "Metric 5", "Metric 6"))

# Add custom CSS to style buttons
st.markdown(
    """
    <style>
    .stButton > button {
        border-radius: 10px;
        background-color: green;
        padding: 10px;
        color: white;
        width: 100%;
        box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2); 
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create two columns to place the buttons side by side
col1, col2 = st.sidebar.columns(2)

# Initialize a state variable for the analysis
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
# Add 'Run Analysis' button in the sidebar
run_analysis = st.sidebar.button("Run Analysis", disabled=st.session_state.analysis_running, key="run_analysis")


st.title("Metrics")

if run_analysis:
    # Set the analysis running state to True
    st.session_state.analysis_running = True
    if uploaded_file and metrics:
        # Display file name, framework, and selected metrics in the right panel
        st.markdown(f"""
            <p style="color:black; font-weight:bold;">File Uploaded: {uploaded_file.name}</p>
            <p style="color:black; font-weight:bold;">Framework Selected: {framework}</p>
            <p style="color:black; font-weight:bold;">Metrics Selected: {', '.join(metrics)}</p>
            """, unsafe_allow_html=True)
        # Show a spinner while running the analysis
        with st.spinner("Running analysis..."):
            # Simulate a long-running process (e.g., 3 seconds delay)
            import time
            time.sleep(5) 

        available_metrics = {metric: round(random.uniform(0.11, 0.99), 2) for metric in metrics}

        # Organize metrics into rows of 3 per row
        num_columns = 3
        metric_groups = [metrics[i:i + num_columns] for i in range(0, len(metrics), num_columns)]

        # Display the metrics group-wise
        for group in metric_groups:
            columns = st.columns(len(group))  # Create a new set of columns for each group
            for i, metric_name in enumerate(group):
                metric_value = available_metrics[metric_name]
                color = "red" if metric_value < 0.4 else "#FFBF00" if metric_value < 0.6 else "green"  # Use hex for amber

                # Display the metric in the respective column
                with columns[i]:
                    st.markdown(
                        f"""
                        <div class="metric-tile">
                            <p class="metric-name">{metric_name}</p>
                            <p class="metric-value" style="color: {color};">{metric_value}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        # Once analysis is complete, enable the Save Results button
        st.session_state.analysis_running = False
    else:
        st.error("Please upload a file and select metrics before running analysis.")
else:
    righ_panel_msg = "Please select metrics and press 'Run Analysis' to see results."
    st.markdown(f"""
            <p style="color:black; font-weight:bold;">{righ_panel_msg}</p>""", unsafe_allow_html=True)

# Add Save Results button, only show if analysis is completed
if not st.session_state.analysis_running:
    # Add 'Save Results' button in the second column
    st.sidebar.button("Save Results", key="save_results")