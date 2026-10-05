"""
Streamlit dashboard for GlucoTwin.

Run with:
    streamlit run scripts/run_dashboard.py

Demonstrates:
- Current patient state
- Personalized Digital Twin trajectory
- Hybrid risk prediction
- What-If counterfactual simulation
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from src.glucotwin.demo.fixture import generate_demo_records, get_demo_summary
from src.glucotwin.demo.app_logic import run_demo_inference, run_whatif_simulation, DISCLAIMER
from src.glucotwin.counterfactual.scenarios import CounterfactualScenario

st.set_page_config(page_title="GlucoTwin Dashboard", page_icon="📈", layout="wide")


@st.cache_data
def load_data():
    """Load deterministic demo records."""
    records = generate_demo_records(n_hours=6.0)
    summary = get_demo_summary(records)
    return records, summary


def main():
    st.title("GlucoTwin Dashboard")
    st.markdown(
        "An AI-powered Digital Twin for Type 1 Diabetes that combines continuous glucose, "
        "insulin and meal data with personalized physiological modeling to forecast near-term "
        "hypoglycemia risk and explore hypothetical future scenarios."
    )
    st.warning(DISCLAIMER)

    # 1. Load Data
    with st.spinner("Loading demo data..."):
        records, summary = load_data()

    # Sidebar: Data Info & Limitations
    st.sidebar.header("Demo Data")
    st.sidebar.text(f"Subject: {summary['subject_id']}")
    st.sidebar.text(f"Duration: {summary['duration_hours']} hours")
    st.sidebar.text(f"Records: {summary['n_records']}")
    st.sidebar.caption(summary['note'])

    st.sidebar.header("Limitations")
    st.sidebar.markdown(
        """
        - CGM units not independently confirmed
        - Insulin/carb unit provenance unresolved
        - Dataset timestamp anomaly unresolved
        - Observational research dataset
        - No clinical validation
        - Research prototype
        - Uncertainty not clinically calibrated
        - What-If simulation is exploratory
        - Not medical advice
        - Not a dosing recommendation
        """
    )

    # Main inference
    try:
        inf_result = run_demo_inference(records)
    except Exception as e:
        st.error(f"Inference error: {e}")
        return

    # 2. Current Patient State
    st.header("Current State")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Glucose", f"{inf_result['last_glucose']} mg/dL")
    with col2:
        st.metric("Data Quality", f"{inf_result['data_quality']['cgm_availability_pct']}% CGM")
    with col3:
        st.metric("Twin Personalized", str(inf_result['twin_params']['is_personalized']))

    # 3. Personalized Digital Twin & 4. Predicted Trajectory
    st.header("Digital Twin Trajectory")
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # History
    hist_t = [ts for ts, g in inf_result['glucose_history']]
    hist_g = [g for ts, g in inf_result['glucose_history']]
    ax.plot(hist_t, hist_g, marker='o', linestyle='-', color='blue', label='Observed Glucose')
    
    # Forecasts
    last_t = hist_t[-1] if hist_t else datetime.now()
    
    traj30_t = [last_t + timedelta(minutes=off) for off, g in inf_result['trajectory_30m']]
    traj30_g = [g for off, g in inf_result['trajectory_30m']]
    ax.plot(traj30_t, traj30_g, marker='.', linestyle='--', color='orange', label='30m Forecast')
    
    traj60_t = [last_t + timedelta(minutes=off) for off, g in inf_result['trajectory_60m']]
    traj60_g = [g for off, g in inf_result['trajectory_60m']]
    # Plot only the part of 60m that comes after 30m
    traj60_t_ext = traj60_t[len(traj30_t)-1:] if traj30_t else traj60_t
    traj60_g_ext = traj60_g[len(traj30_g)-1:] if traj30_g else traj60_g
    ax.plot(traj60_t_ext, traj60_g_ext, marker='.', linestyle='--', color='red', label='60m Forecast')
    
    # Formatting
    ax.axhspan(40, 70, color='red', alpha=0.1, label='Hypoglycemia Zone')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    ax.set_ylabel("Glucose")
    ax.set_xlabel("Time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

    # 5. Hypoglycemia Risk & 6. Explainability
    st.header("Hypoglycemia Risk (Research Estimate)")
    st.info("Models not loaded in this lightweight demo fixture; showing Twin projection context.")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader("30-Minute Risk")
        min_30 = min(traj30_g) if traj30_g else None
        if min_30 and min_30 < 70:
            st.error(f"High risk indicated by Twin trajectory (min {min_30:.1f})")
        else:
            st.success(f"Low risk indicated by Twin trajectory (min {min_30:.1f})")
    with col_r2:
        st.subheader("60-Minute Risk")
        min_60 = min(traj60_g) if traj60_g else None
        if min_60 and min_60 < 70:
            st.error(f"High risk indicated by Twin trajectory (min {min_60:.1f})")
        else:
            st.success(f"Low risk indicated by Twin trajectory (min {min_60:.1f})")

    # 7. Twin-Aware Explainability (V2)
    st.header("Explainability (V2)")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.subheader("Current State")
        st.metric("Insulin Action", f"{inf_result['twin_state']['insulin_action']:.3f}")
        st.metric("Meal State", f"{inf_result['twin_state']['meal_state']:.3f}")
    with col_e2:
        st.subheader("Twin Projection")
        st.metric("Trajectory Area", f"{inf_result['twin_features']['twin_traj_area_60m']:.1f}" if inf_result['twin_features']['twin_traj_area_60m'] else "N/A")
        st.metric("Baseline Deviation", f"{inf_result['twin_features']['twin_baseline_deviation']:.1f}" if inf_result['twin_features']['twin_baseline_deviation'] else "N/A")
    with col_e3:
        st.subheader("ML Contribution")
        st.info("Feature contributions are correlations, not causal proof.")
        st.write("- Twin Baseline Deviation")
        st.write("- Trajectory Minimum")
        st.write("- Insulin Action Trend")

    # 8. What-If Simulation
    st.header("What-If Simulation")
    st.write("Simulate hypothetical scenarios safely. Does NOT constitute medical advice.")
    
    st.write("Select scenarios to compare against the baseline:")
    c1, c2, c3 = st.columns(3)
    with c1:
        run_meal_10 = st.checkbox("+10g Carbs at +5m", value=True)
    with c2:
        run_meal_20 = st.checkbox("+20g Carbs at +5m", value=False)
    with c3:
        run_ins_1 = st.checkbox("+2U Bolus at +5m", value=False)
        
    scenarios = []
    if run_meal_10:
        scenarios.append(CounterfactualScenario.meal_perturbation(10.0, 5.0))
    if run_meal_20:
        scenarios.append(CounterfactualScenario.meal_perturbation(20.0, 5.0))
    if run_ins_1:
        scenarios.append(CounterfactualScenario.insulin_timing(2.0, 5.0))
        
    if scenarios:
        with st.spinner("Simulating..."):
            fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
            
            # Baseline (compute once)
            sim_res = run_whatif_simulation(records, scenarios[0])
            bl_t = [last_t + timedelta(minutes=off) for off, _ in sim_res['baseline']]
            bl_g = [g for _, g in sim_res['baseline']]
            ax_sim.plot(bl_t, bl_g, 'k--', label="Baseline")
            
            # Counterfactuals
            colors = ['b', 'g', 'm', 'c']
            for i, scenario in enumerate(scenarios):
                res = run_whatif_simulation(records, scenario)
                cf_g = [g for _, g in res['counterfactual']]
                ax_sim.plot(bl_t, cf_g, color=colors[i % len(colors)], label=res['scenario'])
            
            ax_sim.axhspan(40, 70, color='red', alpha=0.1)
            ax_sim.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.xticks(rotation=45)
            ax_sim.set_ylabel("Glucose")
            ax_sim.set_xlabel("Time")
            ax_sim.legend(loc="upper left")
            ax_sim.grid(True, alpha=0.3)
            
            st.pyplot(fig_sim)
            st.caption("Simulation only — not a treatment recommendation.")


if __name__ == "__main__":
    main()
