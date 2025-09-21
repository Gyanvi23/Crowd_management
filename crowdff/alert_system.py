import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

class AlertSystem:
    """Advanced alert system for crowd management with multiple alert types and severity levels"""
    
    def __init__(self):
        """Initialize the alert system"""
        # Alert configuration
        self.alert_config = {
            'crowd_density': {
                'enabled': True,
                'moderate_threshold': 10,
                'high_threshold': 20,
                'critical_threshold': 30
            },
            'rapid_increase': {
                'enabled': True,
                'increase_rate_threshold': 5,  # people per frame
                'time_window': 10  # frames
            },
            'hotspot_persistence': {
                'enabled': True,
                'persistence_threshold': 15,  # frames
                'severity_escalation': 5  # frames for severity escalation
            },
            'zone_overflow': {
                'enabled': True,
                'overflow_percentage': 0.8  # 80% of grid filled
            }
        }
        
        # Alert tracking
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)  # Keep last 1000 alerts
        self.alert_counters = defaultdict(int)
        
        # Crowd tracking for trend analysis
        self.crowd_trend_window = deque(maxlen=30)  # Track last 30 frames
        self.zone_trend_tracking = defaultdict(lambda: deque(maxlen=20))
        
        # Cooldown periods to prevent alert spam
        self.alert_cooldowns = defaultdict(float)
        self.cooldown_periods = {
            'moderate_density': 5.0,  # seconds
            'high_density': 3.0,
            'critical_density': 1.0,
            'rapid_increase': 10.0,
            'hotspot_persistent': 15.0,
            'zone_overflow': 8.0
        }
    
    def check_alerts(self, density_data: Dict, normal_threshold: int, 
                    high_threshold: int) -> List[Dict]:
        """
        Check for various types of alerts based on crowd density data
        
        Args:
            density_data: Density analysis results
            normal_threshold: Normal density threshold
            high_threshold: High density threshold
            
        Returns:
            List of active alerts
        """
        current_time = datetime.now()
        alerts = []
        
        # Update tracking data
        self._update_tracking_data(density_data)
        
        # Check density-based alerts
        density_alerts = self._check_density_alerts(density_data, normal_threshold, high_threshold)
        alerts.extend(density_alerts)
        
        # Check rapid increase alerts
        rapid_increase_alerts = self._check_rapid_increase_alerts(density_data)
        alerts.extend(rapid_increase_alerts)
        
        # Check hotspot persistence alerts
        hotspot_alerts = self._check_hotspot_persistence_alerts(density_data)
        alerts.extend(hotspot_alerts)
        
        # Check zone overflow alerts
        overflow_alerts = self._check_zone_overflow_alerts(density_data)
        alerts.extend(overflow_alerts)
        
        # Process and deduplicate alerts
        processed_alerts = self._process_alerts(alerts, current_time)
        
        # Update alert history
        for alert in processed_alerts:
            alert['timestamp'] = current_time
            self.alert_history.append(alert.copy())
            self.alert_counters[alert['type']] += 1
        
        return processed_alerts
    
    def _update_tracking_data(self, density_data: Dict):
        """Update internal tracking data for trend analysis"""
        # Track overall crowd count
        total_people = density_data['metrics']['total_people']
        self.crowd_trend_window.append(total_people)
        
        # Track individual zone densities
        zone_densities = density_data['zone_densities']
        for i, density in enumerate(zone_densities):
            zone_id = f"zone_{i}"
            self.zone_trend_tracking[zone_id].append(density)
    
    def _check_density_alerts(self, density_data: Dict, normal_threshold: int, 
                            high_threshold: int) -> List[Dict]:
        """Check for density-based alerts"""
        alerts = []
        
        if not self.alert_config['crowd_density']['enabled']:
            return alerts
        
        zone_densities = density_data['zone_densities']
        
        for i, density in enumerate(zone_densities):
            zone_id = f"zone_{i // density_data['grid_dimensions'][1]}_{i % density_data['grid_dimensions'][1]}"
            
            # Critical density alert
            if density >= self.alert_config['crowd_density']['critical_threshold']:
                alerts.append({
                    'type': 'critical_density',
                    'severity': 'critical',
                    'message': f"CRITICAL: Extremely high crowd density in {zone_id} ({int(density)} people)",
                    'zone': zone_id,
                    'density': density,
                    'threshold': self.alert_config['crowd_density']['critical_threshold'],
                    'priority': 1
                })
            
            # High density alert
            elif density >= high_threshold:
                alerts.append({
                    'type': 'high_density',
                    'severity': 'high',
                    'message': f"HIGH RISK: Dangerous crowd density in {zone_id} ({int(density)} people)",
                    'zone': zone_id,
                    'density': density,
                    'threshold': high_threshold,
                    'priority': 2
                })
            
            # Moderate density alert
            elif density >= normal_threshold:
                alerts.append({
                    'type': 'moderate_density',
                    'severity': 'moderate',
                    'message': f"MODERATE: Elevated crowd density in {zone_id} ({int(density)} people)",
                    'zone': zone_id,
                    'density': density,
                    'threshold': normal_threshold,
                    'priority': 3
                })
        
        return alerts
    
    def _check_rapid_increase_alerts(self, density_data: Dict) -> List[Dict]:
        """Check for rapid crowd increase alerts"""
        alerts = []
        
        if not self.alert_config['rapid_increase']['enabled'] or len(self.crowd_trend_window) < 5:
            return alerts
        
        # Calculate rate of change
        recent_counts = list(self.crowd_trend_window)[-5:]  # Last 5 frames
        if len(recent_counts) >= 2:
            rate_of_change = recent_counts[-1] - recent_counts[0]  # Change over 5 frames
            
            if rate_of_change >= self.alert_config['rapid_increase']['increase_rate_threshold']:
                alerts.append({
                    'type': 'rapid_increase',
                    'severity': 'high',
                    'message': f"RAPID INCREASE: Crowd growing rapidly (+{int(rate_of_change)} people in recent frames)",
                    'rate_of_change': rate_of_change,
                    'current_count': recent_counts[-1],
                    'priority': 2
                })
        
        return alerts
    
    def _check_hotspot_persistence_alerts(self, density_data: Dict) -> List[Dict]:
        """Check for persistent hotspot alerts"""
        alerts = []
        
        if not self.alert_config['hotspot_persistence']['enabled']:
            return alerts
        
        hotspots = density_data['hotspots']
        
        for hotspot in hotspots:
            persistence = hotspot['persistence']
            hotspot_id = hotspot['id']
            
            # Critical persistence
            if persistence >= self.alert_config['hotspot_persistence']['persistence_threshold']:
                alerts.append({
                    'type': 'hotspot_persistent',
                    'severity': 'critical',
                    'message': f"PERSISTENT HOTSPOT: {hotspot_id} has been overcrowded for {persistence} frames",
                    'zone': hotspot_id,
                    'persistence': persistence,
                    'density': hotspot['density'],
                    'priority': 1
                })
            
            # High persistence warning
            elif persistence >= self.alert_config['hotspot_persistence']['severity_escalation']:
                alerts.append({
                    'type': 'hotspot_developing',
                    'severity': 'high',
                    'message': f"DEVELOPING HOTSPOT: {hotspot_id} showing sustained high density ({persistence} frames)",
                    'zone': hotspot_id,
                    'persistence': persistence,
                    'density': hotspot['density'],
                    'priority': 2
                })
        
        return alerts
    
    def _check_zone_overflow_alerts(self, density_data: Dict) -> List[Dict]:
        """Check for zone overflow alerts (too many zones occupied)"""
        alerts = []
        
        if not self.alert_config['zone_overflow']['enabled']:
            return alerts
        
        total_zones = density_data['grid_dimensions'][0] * density_data['grid_dimensions'][1]
        active_zones = density_data['active_zones']
        occupancy_rate = active_zones / total_zones
        
        if occupancy_rate >= self.alert_config['zone_overflow']['overflow_percentage']:
            alerts.append({
                'type': 'zone_overflow',
                'severity': 'high',
                'message': f"AREA OVERFLOW: {occupancy_rate:.1%} of monitoring zones are occupied ({active_zones}/{total_zones})",
                'occupancy_rate': occupancy_rate,
                'active_zones': active_zones,
                'total_zones': total_zones,
                'priority': 2
            })
        
        return alerts
    
    def _process_alerts(self, alerts: List[Dict], current_time: datetime) -> List[Dict]:
        """Process alerts by applying cooldowns and deduplication"""
        processed_alerts = []
        
        for alert in alerts:
            alert_key = f"{alert['type']}_{alert.get('zone', 'global')}"
            
            # Check cooldown
            cooldown_period = self.cooldown_periods.get(alert['type'], 5.0)
            last_alert_time = self.alert_cooldowns.get(alert_key, 0)
            
            if (current_time.timestamp() - last_alert_time) >= cooldown_period:
                processed_alerts.append(alert)
                self.alert_cooldowns[alert_key] = current_time.timestamp()
        
        # Sort by priority (lower number = higher priority)
        processed_alerts.sort(key=lambda x: x.get('priority', 999))
        
        return processed_alerts
    
    def get_alert_statistics(self) -> Dict:
        """Get statistics about alert activity"""
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'alerts_by_type': {},
                'alerts_by_severity': {},
                'recent_alert_rate': 0.0
            }
        
        # Count alerts by type and severity
        alerts_by_type = defaultdict(int)
        alerts_by_severity = defaultdict(int)
        
        for alert in self.alert_history:
            alerts_by_type[alert['type']] += 1
            alerts_by_severity[alert['severity']] += 1
        
        # Calculate recent alert rate (last 10 minutes)
        recent_cutoff = datetime.now() - timedelta(minutes=10)
        recent_alerts = [a for a in self.alert_history if a['timestamp'] > recent_cutoff]
        recent_alert_rate = len(recent_alerts) / 10.0  # alerts per minute
        
        return {
            'total_alerts': len(self.alert_history),
            'alerts_by_type': dict(alerts_by_type),
            'alerts_by_severity': dict(alerts_by_severity),
            'recent_alert_rate': recent_alert_rate,
            'active_alert_types': len(set(a['type'] for a in recent_alerts))
        }
    
    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alerts for display"""
        recent_alerts = list(self.alert_history)[-limit:]
        
        # Format for display
        formatted_alerts = []
        for alert in recent_alerts:
            formatted_alert = {
                'timestamp': alert['timestamp'].strftime('%H:%M:%S'),
                'type': alert['type'].replace('_', ' ').title(),
                'severity': alert['severity'],
                'message': alert['message'],
                'zone': alert.get('zone', 'N/A')
            }
            formatted_alerts.append(formatted_alert)
        
        return formatted_alerts
    
    def update_alert_config(self, config_updates: Dict):
        """Update alert configuration"""
        for category, settings in config_updates.items():
            if category in self.alert_config:
                self.alert_config[category].update(settings)
    
    def reset_alert_system(self):
        """Reset the alert system state"""
        self.active_alerts.clear()
        self.alert_history.clear()
        self.alert_counters.clear()
        self.crowd_trend_window.clear()
        self.zone_trend_tracking.clear()
        self.alert_cooldowns.clear()
    
    def export_alert_history(self) -> str:
        """Export alert history as JSON string"""
        export_data = {
            'alert_history': [
                {
                    **alert,
                    'timestamp': alert['timestamp'].isoformat()
                }
                for alert in self.alert_history
            ],
            'statistics': self.get_alert_statistics(),
            'configuration': self.alert_config
        }
        
        return json.dumps(export_data, indent=2)
