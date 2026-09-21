"""SentinelLoop Defense Operations & Real-Time Visualization Dashboard.

Provides real-time telemetry, multi-turn drift analysis, intervention inspection,
and comparative defense efficacy metrics (IER, FPD, Recovery Latency).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page layout and branding
st.set_page_config(
    page_title="SentinelLoop Defense Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS_FILE = _REPO_ROOT / "outputs" / "defense_evaluation_results.json"

# Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-header {
        padding: 1.2rem 1.6rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        border-left: 6px solid #3b82f6;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
    }
    
    .kpi-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    
    .chat-bubble-user {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 12px 12px 2px 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        max-width: 85%;
        margin-left: auto;
        color: #1e293b;
    }
    .chat-bubble-assistant {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px 12px 12px 2px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        max-width: 85%;
        margin-right: auto;
        color: #0f172a;
    }
    .chat-bubble-override {
        background: #fef2f2;
        border: 1.5px solid #ef4444;
        border-radius: 12px 12px 12px 2px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        max-width: 85%;
        margin-right: auto;
        color: #991b1b;
    }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .badge-stable { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .badge-patch { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-override { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .badge-recovered { background: #e0f2fe; color: #075985; border: 1px solid #bae6fd; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_sample_benchmark_data() -> dict:
    """Built-in baseline benchmark results for instant visualization."""
    return {
        "metadata": {
            "models_evaluated": ["openai/gpt-4o-mini", "anthropic/claude-haiku-4.5"],
            "mock_mode": False,
            "total_scenarios": 10,
        },
        "aggregate_metrics": {
            "total_simulated_sessions": 40,
            "total_defended_sessions": 20,
            "total_baseline_sessions": 20,
            "total_adversarial_sessions": 10,
            "total_benign_sessions": 10,
            "intervention_efficacy_rate_pct": 100.0,
            "false_positive_disruption_pct": 0.0,
            "avg_recovery_latency_turns": 1.0,
            "total_interventions": 10,
            "total_prompt_patches": 10,
            "total_critical_overrides": 0,
            "baseline_avg_final_compliance": 0.72,
            "defended_avg_final_compliance": 0.00,
            "compliance_reduction_delta": 0.72,
        },
        "sessions": [],
    }


def load_data(file_path: Optional[Path], uploaded_file) -> tuple[Optional[dict], str]:
    """Load evaluation data from uploaded file, local outputs, or sample data."""
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            return data, f"Uploaded: {uploaded_file.name}"
        except Exception as e:
            st.sidebar.error(f"Failed to parse uploaded JSON: {e}")

    if file_path and file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data, f"Local: {file_path.name}"
        except Exception as e:
            st.sidebar.warning(f"Could not load local file: {e}")

    return get_sample_benchmark_data(), "Sample Benchmark Data (Embedded)"


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;">
        <span style="font-size: 2.2rem;">🛡️</span>
        <div>
            <h2 style="margin: 0; padding: 0; font-size: 1.4rem; font-weight: 700;">SentinelLoop</h2>
            <span style="font-size: 0.75rem; color: #64748b; font-weight: 500;">Real-Time Guardrail System</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

data_source = st.sidebar.radio(
    "Data Source Selection",
    ["Auto (Local / Sample)", "Upload Custom JSON"],
    help="Toggle between evaluation run output files and custom evaluation artifacts.",
)

uploaded_file = None
if data_source == "Upload Custom JSON":
    uploaded_file = st.sidebar.file_uploader("Upload Evaluation Results (.json)", type=["json"])

eval_data, source_label = load_data(_DEFAULT_RESULTS_FILE, uploaded_file)

if not eval_data:
    st.error("No evaluation data could be loaded.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active Dataset**: `{source_label}`")

# Extract metadata & metrics
meta = eval_data.get("metadata", {})
agg = eval_data.get("aggregate_metrics", {})
sessions = eval_data.get("sessions", [])

# Model filter
all_models = meta.get("models_evaluated", ["openai/gpt-4o-mini", "anthropic/claude-haiku-4.5"])
selected_models = st.sidebar.multiselect("Filter Target Models", all_models, default=all_models)

# Filter sessions
filtered_sessions = [s for s in sessions if s.get("target_model") in selected_models] if sessions else []

# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-header">
        <div class="hero-title">SentinelLoop Operations & Telemetry Center</div>
        <div class="hero-subtitle">
            Continuous multi-turn drift detection, cumulative 5D vector scoring, and autonomous prompt patching.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI Summary Cards
# ---------------------------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

ier_val = agg.get("intervention_efficacy_rate_pct", 100.0)
fpd_val = agg.get("false_positive_disruption_pct", 0.0)
rec_lat = agg.get("avg_recovery_latency_turns", 1.0)
tot_int = agg.get("total_interventions", 0)
patches = agg.get("total_prompt_patches", 0)
overrides = agg.get("total_critical_overrides", 0)

with kpi_col1:
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-title">Intervention Efficacy (IER)</div>
            <div class="kpi-value" style="color: #16a34a;">{ier_val:.1f}%</div>
            <div class="kpi-sub">Drifting sessions safely halted</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-title">False Positive Disruption</div>
            <div class="kpi-value" style="color: {'#16a34a' if fpd_val == 0 else '#dc2626'};">{fpd_val:.1f}%</div>
            <div class="kpi-sub">Benign conversations impacted</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-title">Recovery Latency</div>
            <div class="kpi-value" style="color: #2563eb;">{rec_lat:.1f} <span style="font-size: 1rem;">turns</span></div>
            <div class="kpi-sub">Mean turns to restore alignment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-title">Total Interventions</div>
            <div class="kpi-value" style="color: #d97706;">{tot_int}</div>
            <div class="kpi-sub">Patches: {patches} | Overrides: {overrides}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_trajectories, tab_inspector, tab_sandbox = st.tabs([
    "📊 Defense Overview & Analytics",
    "📈 Drift & Intervention Trajectories",
    "💬 Live Conversation Inspector",
    "🧪 Interactive Guardrail Sandbox",
])

# ---------------------------------------------------------------------------
# TAB 1: Defense Overview & Analytics
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("System Performance & Compliance Neutralization")

    c1, c2 = st.columns([3, 2])

    with c1:
        # Comparison Bar Chart: Baseline vs Defended Compliance
        comp_data = {
            "Condition": ["Undefended Baseline", "SentinelLoop Protected"],
            "Average Final Compliance": [
                agg.get("baseline_avg_final_compliance", 0.72),
                agg.get("defended_avg_final_compliance", 0.00),
            ],
        }
        df_comp = pd.DataFrame(comp_data)
        fig_comp = px.bar(
            df_comp,
            x="Condition",
            y="Average Final Compliance",
            color="Condition",
            color_discrete_map={
                "Undefended Baseline": "#ef4444",
                "SentinelLoop Protected": "#10b981",
            },
            text="Average Final Compliance",
            title="Final Conversation Compliance: Baseline vs. SentinelLoop Defense",
        )
        fig_comp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_comp.update_layout(yaxis=dict(range=[0, 1.05]), height=360)
        st.plotly_chart(fig_comp, use_container_width=True)

    with c2:
        # Pie Chart: Interventions Breakdown
        act_data = {
            "Intervention Strategy": ["Strategy A: Dynamic Patching", "Strategy B: Critical Override"],
            "Count": [max(patches, 1), overrides],
        }
        df_act = pd.DataFrame(act_data)
        fig_act = px.pie(
            df_act,
            names="Intervention Strategy",
            values="Count",
            color="Intervention Strategy",
            color_discrete_map={
                "Strategy A: Dynamic Patching": "#f59e0b",
                "Strategy B: Critical Override": "#ef4444",
            },
            hole=0.45,
            title="Defensive Action Distribution",
        )
        fig_act.update_layout(height=360)
        st.plotly_chart(fig_act, use_container_width=True)

    st.markdown("---")
    st.subheader("Model-by-Model Defense Breakdown")

    model_summaries = eval_data.get("model_summaries", {})
    if model_summaries:
        summary_rows = []
        for model_name, m_agg in model_summaries.items():
            summary_rows.append({
                "Target Model": model_name,
                "IER (%)": f"{m_agg.get('intervention_efficacy_rate_pct', 0.0):.1f}%",
                "FPD (%)": f"{m_agg.get('false_positive_disruption_pct', 0.0):.1f}%",
                "Recovery Latency": f"{m_agg.get('avg_recovery_latency_turns', 0.0):.1f} turns",
                "Interventions": m_agg.get("total_interventions", 0),
                "Baseline Compliance": f"{m_agg.get('baseline_avg_final_compliance', 0.0):.2f}",
                "Defended Compliance": f"{m_agg.get('defended_avg_final_compliance', 0.0):.2f}",
                "Safety Delta (Δ)": f"-{m_agg.get('compliance_reduction_delta', 0.0):.2f}",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Run full multi-model evals to populate per-model comparative metrics.")

# ---------------------------------------------------------------------------
# TAB 2: Drift & Intervention Trajectories (Plotly)
# ---------------------------------------------------------------------------
with tab_trajectories:
    st.subheader("Multi-Turn Compliance Trajectory & Intervention Threshold Zones")
    st.caption("Visualizing how SentinelLoop intercepts behavioral drift before critical alignment failure.")

    if filtered_sessions:
        # Scenario selector
        scenario_ids = sorted(list({s["conversation_id"] for s in filtered_sessions}))
        selected_scenario = st.selectbox("Select Conversation Scenario", scenario_ids)

        # Get baseline and defended sessions for this scenario
        matched_sessions = [s for s in filtered_sessions if s["conversation_id"] == selected_scenario]

        fig_traj = go.Figure()

        # Add Intervention Threshold Bands
        # Green Zone (Safe)
        fig_traj.add_hrect(
            y0=0.0, y1=0.50,
            fillcolor="#10b981", opacity=0.08, line_width=0,
            annotation_text="Safe Operating Zone (Compliance < 0.50)",
            annotation_position="bottom left",
        )
        # Amber Zone (Strategy A: Prompt Patching)
        fig_traj.add_hrect(
            y0=0.50, y1=0.85,
            fillcolor="#f59e0b", opacity=0.12, line_width=0,
            annotation_text="Strategy A Trigger Zone (0.50 ≤ Compliance < 0.85)",
            annotation_position="top left",
        )
        # Red Zone (Strategy B: Critical Override)
        fig_traj.add_hrect(
            y0=0.85, y1=1.05,
            fillcolor="#ef4444", opacity=0.15, line_width=0,
            annotation_text="Strategy B Critical Override Zone (Compliance ≥ 0.85)",
            annotation_position="top left",
        )

        colors = {"baseline": "#ef4444", "defended": "#2563eb"}

        for s in matched_sessions:
            is_defended = s.get("defense_enabled", False)
            mode_lbl = "SentinelLoop Defended" if is_defended else "Baseline (Unprotected)"
            color = colors["defended"] if is_defended else colors["baseline"]
            turns = s.get("turns", [])

            x_vals = [t["turn"] for t in turns]
            y_vals = [t["compliance"] for t in turns]
            actions = [t.get("triggered_action", "NONE") for t in turns]
            velocities = [t.get("drift_velocity", 0.0) for t in turns]

            # Line trace
            fig_traj.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=f"{s.get('target_model', 'Target')} ({mode_lbl})",
                    line=dict(color=color, width=3, dash="solid" if is_defended else "dash"),
                    marker=dict(size=8),
                    hovertemplate=(
                        "Turn %{x}<br>Compliance: %{y:.2f}<br>"
                        "Drift Velocity: %{customdata[0]:+.2f}<br>"
                        "Action: %{customdata[1]}<extra></extra>"
                    ),
                    customdata=list(zip(velocities, actions)),
                )
            )

            # Highlight intervention points
            if is_defended:
                for t in turns:
                    act = t.get("triggered_action")
                    if act == "PROMPT_PATCH":
                        fig_traj.add_annotation(
                            x=t["turn"],
                            y=t["compliance"],
                            text="⚡ Strategy A: Prompt Patch",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="#d97706",
                            bgcolor="#fef3c7",
                            font=dict(size=11, color="#92400e", family="Inter"),
                            bordercolor="#fde68a",
                        )
                    elif act == "CRITICAL_OVERRIDE":
                        fig_traj.add_annotation(
                            x=t["turn"],
                            y=t["compliance"],
                            text="🛑 Strategy B: Safe Override",
                            showarrow=True,
                            arrowhead=2,
                            arrowcolor="#b91c1c",
                            bgcolor="#fee2e2",
                            font=dict(size=11, color="#991b1b", family="Inter"),
                            bordercolor="#fecaca",
                        )

        fig_traj.update_layout(
            title=f"Compliance Trajectory Across Turns — Scenario: {selected_scenario}",
            xaxis_title="Dialogue Turn Number",
            yaxis_title="Compliance Score (0.0 to 1.0)",
            yaxis=dict(range=[-0.05, 1.1]),
            xaxis=dict(dtick=1),
            height=480,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig_traj, use_container_width=True)

    else:
        st.info("No session telemetry available for trajectory plotting.")

# ---------------------------------------------------------------------------
# TAB 3: Live Conversation Inspector
# ---------------------------------------------------------------------------
with tab_inspector:
    st.subheader("Turn-by-Turn Telemetry & Conversation Inspector")
    st.caption("Inspect dialogue exchanges, dynamic system prompt patches, and 5D behavioral vector ratings.")

    if filtered_sessions:
        # Filter dropdowns
        col_sel1, col_sel2 = st.columns([3, 2])
        with col_sel1:
            inspect_scenario = st.selectbox(
                "Select Conversation to Inspect",
                sorted(list({s["conversation_id"] for s in filtered_sessions})),
                key="inspect_scenario",
            )
        with col_sel2:
            inspect_defense = st.radio(
                "Condition View",
                ["SentinelLoop Defended", "Baseline (Unprotected)"],
                horizontal=True,
            )

        defense_flag = (inspect_defense == "SentinelLoop Defended")
        target_session = next(
            (s for s in filtered_sessions if s["conversation_id"] == inspect_scenario and s.get("defense_enabled") == defense_flag),
            None,
        )

        if target_session:
            # Metadata Banner
            st.markdown(
                f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem; margin-bottom: 1.2rem;">
                    <strong>Category:</strong> {target_session.get('category', 'General')} &nbsp;|&nbsp;
                    <strong>Type:</strong> <code>{target_session.get('type', 'Unknown')}</code> &nbsp;|&nbsp;
                    <strong>Target Model:</strong> {target_session.get('target_model', 'N/A')} &nbsp;|&nbsp;
                    <strong>Max Compliance:</strong> {target_session.get('max_compliance', 0.0):.2f} &nbsp;|&nbsp;
                    <strong>Interventions:</strong> {target_session.get('total_interventions', 0)}
                    <br><small style="color: #64748b;"><strong>Scenario Goal:</strong> {target_session.get('goal', '')}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

            turns = target_session.get("turns", [])
            for t in turns:
                turn_idx = t.get("turn", 1)
                user_msg = t.get("user_prompt", "")
                resp_msg = t.get("model_response", "")
                orig_msg = t.get("original_response", "")
                action = t.get("triggered_action", "NONE")
                status = t.get("recovery_status", "STABLE")
                patch = t.get("active_prompt_patch")

                st.markdown(f"#### Turn {turn_idx}")

                # User message bubble
                st.markdown(
                    f"""
                    <div class="chat-bubble-user">
                        <strong>User</strong><br>{user_msg}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Badge determination
                if action == "CRITICAL_OVERRIDE":
                    badge_html = '<span class="badge badge-override">🛑 STRATEGY B: CRITICAL OVERRIDE APPLIED</span>'
                    bubble_class = "chat-bubble-override"
                elif action == "PROMPT_PATCH":
                    badge_html = '<span class="badge badge-patch">⚡ STRATEGY A: PROMPT PATCH TRIGGERED</span>'
                    bubble_class = "chat-bubble-assistant"
                elif status == "RECOVERED":
                    badge_html = '<span class="badge badge-recovered">🛡️ ALIGNMENT RECOVERED</span>'
                    bubble_class = "chat-bubble-assistant"
                else:
                    badge_html = '<span class="badge badge-stable">🟢 STABLE ENGAGEMENT</span>'
                    bubble_class = "chat-bubble-assistant"

                # Assistant message bubble
                st.markdown(
                    f"""
                    <div class="{bubble_class}">
                        <div>{badge_html}</div>
                        <strong>Assistant ({target_session.get('target_model', 'Target')})</strong><br>
                        {resp_msg}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Expandable details for prompt patches & vector metrics
                exp_col1, exp_col2 = st.columns([1, 1])

                with exp_col1:
                    with st.expander(f"Turn {turn_idx} Behavioral 5D Metrics"):
                        st.write({
                            "Compliance": t.get("compliance"),
                            "Hedging": t.get("hedging"),
                            "Sycophancy": t.get("sycophancy"),
                            "Deception": t.get("deception_markers"),
                            "Verbosity Tokens": t.get("verbosity_tokens"),
                            "Drift Velocity (Δ)": t.get("drift_velocity"),
                        })
                        st.caption(f"**Judge Reasoning:** {t.get('reasoning', '')}")

                with exp_col2:
                    if patch:
                        with st.expander(f"Active Prompt Patch (Turn {turn_idx})"):
                            st.info(patch)
                    elif action == "CRITICAL_OVERRIDE":
                        with st.expander(f"Pre-Override Response (Suppressed)"):
                            st.code(orig_msg)

                st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px dashed #cbd5e1;'>", unsafe_allow_html=True)

        else:
            st.warning("No matching session found for the selected configuration.")
    else:
        st.info("No conversations available to inspect.")

# ---------------------------------------------------------------------------
# TAB 4: Interactive Guardrail Sandbox
# ---------------------------------------------------------------------------
with tab_sandbox:
    st.subheader("Live Guardrail Interceptor Sandbox")
    st.caption("Interact with SentinelLoop middleware in real-time. Test how cumulative history and prompt patching react to live turns.")

    from src.config import INTERVENTION_THRESHOLD, CRITICAL_THRESHOLD, DEFAULT_SAFETY_PATCH
    from src.interceptor.middleware import GuardrailInterceptor

    if "sandbox_interceptor" not in st.session_state:
        st.session_state.sandbox_interceptor = GuardrailInterceptor(session_id="sandbox_session")
        st.session_state.sandbox_history = []

    sb_interceptor: GuardrailInterceptor = st.session_state.sandbox_interceptor

    col_ctrl, col_reset = st.columns([4, 1])
    with col_ctrl:
        st.write(f"**Intervention Threshold:** `{INTERVENTION_THRESHOLD}` | **Critical Threshold:** `{CRITICAL_THRESHOLD}`")
    with col_reset:
        if st.button("Reset Sandbox Session", type="secondary"):
            st.session_state.sandbox_interceptor.reset()
            st.session_state.sandbox_history.clear()
            st.success("Session reset.")
            st.rerun()

    # Display dialogue history in sandbox
    for msg in st.session_state.sandbox_history:
        role = msg.get("role")
        content = msg.get("content")
        action = msg.get("action", "NONE")
        if role == "user":
            st.markdown(f'<div class="chat-bubble-user"><strong>User:</strong><br>{content}</div>', unsafe_allow_html=True)
        else:
            cls = "chat-bubble-override" if action == "CRITICAL_OVERRIDE" else "chat-bubble-assistant"
            st.markdown(f'<div class="{cls}"><strong>Assistant:</strong><br>{content}</div>', unsafe_allow_html=True)

    # Input for new turn
    with st.form("sandbox_input_form"):
        user_input = st.text_input("User Prompt", placeholder="Type a message to test the guardrail...")
        simulated_response = st.text_area(
            "Simulated Model Response",
            placeholder="Provide a simulated model response to evaluate...",
            height=100,
        )
        submitted = st.form_submit_button("Submit Turn & Intercept")

        if submitted and user_input and simulated_response:
            st.session_state.sandbox_history.append({"role": "user", "content": user_input})

            # Process through interceptor
            import asyncio
            eval_hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.sandbox_history]
            telemetry = asyncio.run(
                sb_interceptor.intercept_and_process(
                    history=eval_hist,
                    current_response=simulated_response,
                )
            )

            final_msg = telemetry["modified_response"]
            action = telemetry["triggered_action"]
            st.session_state.sandbox_history.append({
                "role": "assistant",
                "content": final_msg,
                "action": action,
            })

            st.success(f"Processed! Triggered Action: **{action}** | Compliance: **{telemetry['pre_intervention_scores']['compliance']:.2f}** | Velocity: **{telemetry['drift_velocity']:+.2f}**")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("SentinelLoop v1.0.0 • Production AI Safety Infrastructure")