import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import time
from datetime import datetime, timedelta
import threading
import queue

from crowd_detector import CrowdDetector
from density_analyzer import DensityAnalyzer
from alert_system import AlertSystem
from config import VENUE_CONFIGS, ALERT_THRESHOLDS

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = CrowdDetector()
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = DensityAnalyzer()
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()
if 'crowd_history' not in st.session_state:
    st.session_state.crowd_history = []
if 'processing_video' not in st.session_state:
    st.session_state.processing_video = False

def main():
    st.set_page_config(
        page_title="AI Crowd Management System",
        page_icon="👥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 AI-Powered Crowd Management System")
    st.markdown("**Real-time crowd density monitoring using YOLOv8**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Venue type selection
        venue_type = st.selectbox(
            "Select Venue Type",
            list(VENUE_CONFIGS.keys()),
            help="Choose the type of venue to apply appropriate density thresholds"
        )
        
        venue_config = VENUE_CONFIGS[venue_type]
        st.info(f"**{venue_type}**\n\n"
                f"🟢 Normal: < {venue_config['normal_threshold']} people/zone\n\n"
                f"🟡 Moderate: {venue_config['normal_threshold']}-{venue_config['high_threshold']} people/zone\n\n"
                f"🔴 High Risk: > {venue_config['high_threshold']} people/zone")
        
        # Custom thresholds
        st.subheader("Custom Thresholds")
        normal_threshold = st.slider(
            "Normal Density Threshold",
            1, 50, venue_config['normal_threshold'],
            help="Maximum people per zone for normal density"
        )
        
        high_threshold = st.slider(
            "High Risk Threshold",
            normal_threshold + 1, 100, venue_config['high_threshold'],
            help="Minimum people per zone for high risk alert"
        )
        
        # Grid configuration
        st.subheader("Analysis Grid")
        grid_rows = st.slider("Grid Rows", 3, 20, 10)
        grid_cols = st.slider("Grid Columns", 3, 20, 10)
        
        # Update thresholds
        st.session_state.analyzer.update_thresholds(normal_threshold, high_threshold)
        st.session_state.analyzer.set_grid_size(grid_rows, grid_cols)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Analysis", "📊 Dashboard", "📈 Analytics", "⚠️ Alerts"])
    
    with tab1:
        st.header("Video Analysis")
        
        # Analysis mode selection
        analysis_mode = st.radio(
            "Select Analysis Mode",
            ["Upload Video File", "Webcam Stream (Demo)"],
            horizontal=True
        )
        
        if analysis_mode == "Upload Video File":
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file for crowd analysis"
            )
            
            if uploaded_file is not None:
                process_uploaded_video(uploaded_file, normal_threshold, high_threshold, grid_rows, grid_cols)
        
        else:
            st.info("📹 Webcam streaming would require camera access. For demo purposes, please upload a video file.")
    
    with tab2:
        display_dashboard()
    
    with tab3:
        display_analytics()
    
    with tab4:
        display_alerts()

def process_uploaded_video(uploaded_file, normal_threshold, high_threshold, grid_rows, grid_cols):
    """Process uploaded video file for crowd analysis"""
    
    if st.button("🚀 Start Analysis", type="primary"):
        st.session_state.processing_video = True
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(tmp_file_path)
            
            if not cap.isOpened():
                st.error("❌ Error: Could not open video file")
                return
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            st.info(f"📹 Video Info: {duration:.1f}s, {fps} FPS, {total_frames} frames")
            
            # Create containers for real-time updates
            col1, col2 = st.columns([2, 1])
            
            with col1:
                video_placeholder = st.empty()
                heatmap_placeholder = st.empty()
            
            with col2:
                metrics_placeholder = st.empty()
                alerts_placeholder = st.empty()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            
            while cap.isOpened() and st.session_state.processing_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                detections = st.session_state.detector.detect_people(frame)
                
                # Analyze density
                density_data = st.session_state.analyzer.analyze_frame(frame, detections)
                
                # Generate alerts
                alerts = st.session_state.alert_system.check_alerts(
                    density_data, normal_threshold, high_threshold
                )
                
                # Store history
                timestamp = datetime.now() - timedelta(seconds=(total_frames - frame_count) / fps)
                crowd_data = {
                    'timestamp': timestamp,
                    'total_people': len(detections),
                    'density_data': density_data,
                    'alerts': alerts
                }
                st.session_state.crowd_history.append(crowd_data)
                
                # Update displays
                annotated_frame = draw_detections(frame, detections)
                
                with video_placeholder.container():
                    st.image(annotated_frame, channels="BGR", caption=f"Frame {frame_count + 1}")
                
                with heatmap_placeholder.container():
                    heatmap_fig = create_density_heatmap(density_data, grid_rows, grid_cols)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                
                with metrics_placeholder.container():
                    display_current_metrics(len(detections), density_data, alerts)
                
                with alerts_placeholder.container():
                    display_current_alerts(alerts)
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                
                # Add small delay for visualization
                time.sleep(0.1)
            
            cap.release()
            st.session_state.processing_video = False
            st.success("✅ Video analysis completed!")
            
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            st.session_state.processing_video = False
    
    # Stop button
    if st.session_state.processing_video:
        if st.button("🛑 Stop Analysis", type="secondary"):
            st.session_state.processing_video = False

def draw_detections(frame, detections):
    """Draw bounding boxes around detected people"""
    annotated_frame = frame.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence score
        label = f"Person: {confidence:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw total count
    total_count = len(detections)
    cv2.putText(annotated_frame, f"Total People: {total_count}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return annotated_frame

def create_density_heatmap(density_data, grid_rows, grid_cols):
    """Create crowd density heatmap"""
    density_matrix = np.array(density_data['density_matrix'])
    
    fig = go.Figure(data=go.Heatmap(
        z=density_matrix,
        colorscale='Reds',
        showscale=True,
        colorbar=dict(title="People Count")
    ))
    
    fig.update_layout(
        title="Crowd Density Heatmap",
        xaxis_title="Grid Column",
        yaxis_title="Grid Row",
        height=400,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def display_current_metrics(total_people, density_data, alerts):
    """Display current crowd metrics"""
    st.metric("👥 Total People", total_people)
    st.metric("📊 Active Zones", density_data['active_zones'])
    st.metric("🔥 Hotspots", len([z for z in density_data['zone_densities'] if z > 10]))
    
    if alerts:
        st.metric("⚠️ Active Alerts", len(alerts))

def display_current_alerts(alerts):
    """Display current alerts"""
    st.subheader("🚨 Current Alerts")
    
    if not alerts:
        st.success("✅ No alerts - All zones safe")
    else:
        for alert in alerts:
            if alert['severity'] == 'high':
                st.error(f"🔴 **HIGH RISK**: {alert['message']}")
            elif alert['severity'] == 'moderate':
                st.warning(f"🟡 **MODERATE**: {alert['message']}")

def display_dashboard():
    """Display main dashboard with overview metrics"""
    st.header("📊 Crowd Management Dashboard")
    
    if not st.session_state.crowd_history:
        st.info("🔄 No data available. Please analyze a video first.")
        return
    
    # Latest data
    latest_data = st.session_state.crowd_history[-1] if st.session_state.crowd_history else None
    
    if latest_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "👥 Current Count",
                latest_data['total_people'],
                delta=None
            )
        
        with col2:
            active_zones = latest_data['density_data']['active_zones']
            st.metric("📍 Active Zones", active_zones)
        
        with col3:
            total_alerts = len(latest_data['alerts'])
            st.metric("⚠️ Active Alerts", total_alerts)
        
        with col4:
            max_zone_density = max(latest_data['density_data']['zone_densities']) if latest_data['density_data']['zone_densities'] else 0
            st.metric("🔥 Max Zone Density", max_zone_density)
    
    # Recent activity timeline
    if len(st.session_state.crowd_history) > 1:
        st.subheader("📈 Crowd Timeline")
        
        # Prepare timeline data
        times = [entry['timestamp'] for entry in st.session_state.crowd_history[-50:]]  # Last 50 entries
        people_counts = [entry['total_people'] for entry in st.session_state.crowd_history[-50:]]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=people_counts,
            mode='lines+markers',
            name='People Count',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="People Count Over Time",
            xaxis_title="Time",
            yaxis_title="Number of People",
            height=400,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_analytics():
    """Display detailed analytics and insights"""
    st.header("📈 Analytics & Insights")
    
    if not st.session_state.crowd_history:
        st.info("🔄 No data available. Please analyze a video first.")
        return
    
    # Crowd statistics
    st.subheader("📊 Crowd Statistics")
    
    people_counts = [entry['total_people'] for entry in st.session_state.crowd_history]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📈 Peak Count", max(people_counts) if people_counts else 0)
    
    with col2:
        st.metric("📊 Average Count", int(np.mean(people_counts)) if people_counts else 0)
    
    with col3:
        st.metric("📉 Min Count", min(people_counts) if people_counts else 0)
    
    # Distribution analysis
    if people_counts:
        st.subheader("📊 Crowd Distribution")
        
        fig = px.histogram(
            x=people_counts,
            nbins=20,
            title="Distribution of People Counts",
            labels={'x': 'Number of People', 'y': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Alert frequency analysis
    st.subheader("⚠️ Alert Analysis")
    
    alert_counts = {'high': 0, 'moderate': 0, 'none': 0}
    
    for entry in st.session_state.crowd_history:
        alerts = entry['alerts']
        if not alerts:
            alert_counts['none'] += 1
        else:
            has_high = any(alert['severity'] == 'high' for alert in alerts)
            has_moderate = any(alert['severity'] == 'moderate' for alert in alerts)
            
            if has_high:
                alert_counts['high'] += 1
            elif has_moderate:
                alert_counts['moderate'] += 1
    
    # Alert distribution pie chart
    if sum(alert_counts.values()) > 0:
        fig = px.pie(
            values=list(alert_counts.values()),
            names=['High Risk', 'Moderate Risk', 'Safe'],
            title="Alert Distribution",
            color_discrete_map={
                'High Risk': '#ff4444',
                'Moderate Risk': '#ffaa00',
                'Safe': '#44ff44'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

def display_alerts():
    """Display alert history and management"""
    st.header("⚠️ Alert Management")
    
    if not st.session_state.crowd_history:
        st.info("🔄 No data available. Please analyze a video first.")
        return
    
    # Recent alerts
    st.subheader("🕒 Recent Alerts")
    
    recent_alerts = []
    for entry in st.session_state.crowd_history[-20:]:  # Last 20 entries
        if entry['alerts']:
            for alert in entry['alerts']:
                recent_alerts.append({
                    'timestamp': entry['timestamp'],
                    'severity': alert['severity'],
                    'message': alert['message'],
                    'zone': alert.get('zone', 'Unknown')
                })
    
    if recent_alerts:
        # Convert to DataFrame for better display
        alerts_df = pd.DataFrame(recent_alerts)
        alerts_df['timestamp'] = alerts_df['timestamp'].dt.strftime('%H:%M:%S')
        
        # Color code by severity
        def style_severity(row):
            if row['severity'] == 'high':
                return ['background-color: #ffebee'] * len(row)
            elif row['severity'] == 'moderate':
                return ['background-color: #fff8e1'] * len(row)
            return [''] * len(row)
        
        styled_df = alerts_df.style.apply(style_severity, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No recent alerts - All zones operating safely")
    
    # Alert summary
    st.subheader("📊 Alert Summary")
    
    if recent_alerts:
        severity_counts = pd.Series([alert['severity'] for alert in recent_alerts]).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'high' in severity_counts:
                st.error(f"🔴 High Risk Alerts: {severity_counts['high']}")
            else:
                st.success("🔴 High Risk Alerts: 0")
        
        with col2:
            if 'moderate' in severity_counts:
                st.warning(f"🟡 Moderate Risk Alerts: {severity_counts['moderate']}")
            else:
                st.success("🟡 Moderate Risk Alerts: 0")

if __name__ == "__main__":
    main()
