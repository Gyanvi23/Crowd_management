"""
Configuration settings for the AI Crowd Management System
"""

# Venue-specific configuration templates
VENUE_CONFIGS = {
    "Cricket Stadium": {
        "normal_threshold": 8,
        "high_threshold": 18,
        "description": "Large open venue with multiple entry/exit points",
        "grid_recommendation": {"rows": 12, "cols": 15},
        "special_considerations": [
            "Monitor gate areas during entry/exit",
            "Watch for crowd surges during exciting moments",
            "Pay attention to concession stand areas"
        ]
    },
    
    "Temple/Religious Site": {
        "normal_threshold": 6,
        "high_threshold": 12,
        "description": "Sacred space with potential for sudden crowd movements",
        "grid_recommendation": {"rows": 10, "cols": 10},
        "special_considerations": [
            "Monitor main worship areas",
            "Watch for queue formations",
            "Consider prayer time crowd patterns"
        ]
    },
    
    "Tourist Hotspot": {
        "normal_threshold": 10,
        "high_threshold": 20,
        "description": "Popular destination with varying crowd patterns",
        "grid_recommendation": {"rows": 8, "cols": 12},
        "special_considerations": [
            "Monitor photo-taking spots",
            "Watch for tour group concentrations",
            "Consider seasonal variations"
        ]
    },
    
    "Concert/Event Venue": {
        "normal_threshold": 12,
        "high_threshold": 25,
        "description": "Entertainment venue with high crowd density tolerance",
        "grid_recommendation": {"rows": 15, "cols": 20},
        "special_considerations": [
            "Monitor stage front areas",
            "Watch for mosh pit formations",
            "Pay attention to bar/refreshment areas"
        ]
    },
    
    "Shopping Mall": {
        "normal_threshold": 7,
        "high_threshold": 15,
        "description": "Retail environment with mixed crowd patterns",
        "grid_recommendation": {"rows": 10, "cols": 12},
        "special_considerations": [
            "Monitor store entrances during sales",
            "Watch for food court congestion",
            "Consider escalator and elevator areas"
        ]
    },
    
    "Transportation Hub": {
        "normal_threshold": 15,
        "high_threshold": 30,
        "description": "High-throughput area with rapid crowd movement",
        "grid_recommendation": {"rows": 8, "cols": 16},
        "special_considerations": [
            "Monitor platform areas",
            "Watch for boarding/alighting congestion",
            "Consider rush hour patterns"
        ]
    },
    
    "Festival/Fair Grounds": {
        "normal_threshold": 10,
        "high_threshold": 22,
        "description": "Outdoor event space with varied attractions",
        "grid_recommendation": {"rows": 12, "cols": 18},
        "special_considerations": [
            "Monitor ride queues",
            "Watch for performance area crowds",
            "Consider food vendor concentrations"
        ]
    },
    
    "Custom Venue": {
        "normal_threshold": 8,
        "high_threshold": 16,
        "description": "Customizable settings for specific venue requirements",
        "grid_recommendation": {"rows": 10, "cols": 10},
        "special_considerations": [
            "Adjust thresholds based on venue capacity",
            "Consider specific crowd flow patterns",
            "Monitor high-risk areas identified by venue management"
        ]
    }
}

# Alert severity thresholds
ALERT_THRESHOLDS = {
    "low": {
        "color": "#4CAF50",
        "icon": "✅",
        "action": "Monitor"
    },
    "moderate": {
        "color": "#FF9800",
        "icon": "⚠️",
        "action": "Increased Monitoring"
    },
    "high": {
        "color": "#F44336",
        "icon": "🚨",
        "action": "Immediate Attention Required"
    },
    "critical": {
        "color": "#9C27B0",
        "icon": "🔴",
        "action": "Emergency Response Needed"
    }
}

# YOLOv8 model configurations
YOLO_CONFIGS = {
    "yolov8n.pt": {
        "name": "Nano",
        "description": "Fastest inference, lower accuracy",
        "recommended_for": "Real-time applications, limited hardware"
    },
    "yolov8s.pt": {
        "name": "Small", 
        "description": "Balanced speed and accuracy",
        "recommended_for": "Most applications, good balance"
    },
    "yolov8m.pt": {
        "name": "Medium",
        "description": "Higher accuracy, moderate speed",
        "recommended_for": "Accuracy-focused applications"
    },
    "yolov8l.pt": {
        "name": "Large",
        "description": "High accuracy, slower inference",
        "recommended_for": "Offline analysis, high-end hardware"
    },
    "yolov8x.pt": {
        "name": "Extra Large",
        "description": "Highest accuracy, slowest inference",
        "recommended_for": "Maximum accuracy requirements"
    }
}

# Video processing settings
VIDEO_SETTINGS = {
    "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    "max_file_size_mb": 500,
    "default_fps": 30,
    "processing_fps": 10,  # Process every nth frame for performance
    "max_resolution": (1920, 1080),
    "min_resolution": (640, 480)
}

# Grid analysis settings
GRID_SETTINGS = {
    "min_rows": 3,
    "max_rows": 25,
    "min_cols": 3,
    "max_cols": 25,
    "default_rows": 10,
    "default_cols": 10,
    "recommended_cell_size_pixels": 80  # Minimum recommended cell size
}

# Performance optimization settings
PERFORMANCE_SETTINGS = {
    "confidence_threshold": 0.5,
    "temporal_smoothing": 0.3,
    "max_history_frames": 100,
    "alert_cooldown_seconds": 5,
    "batch_processing_size": 5
}

# UI/Display settings
DISPLAY_SETTINGS = {
    "heatmap_colorscale": "Reds",
    "detection_bbox_color": (0, 255, 0),
    "detection_bbox_thickness": 2,
    "font_scale": 0.7,
    "font_thickness": 2,
    "max_alerts_display": 20
}

# System limits and constraints
SYSTEM_LIMITS = {
    "max_concurrent_streams": 1,  # For this demo implementation
    "max_video_duration_minutes": 30,
    "max_detections_per_frame": 1000,
    "memory_limit_mb": 2048,
    "processing_timeout_seconds": 300
}

# Export and logging settings
EXPORT_SETTINGS = {
    "supported_export_formats": ["json", "csv", "pdf"],
    "default_export_format": "json",
    "include_metadata": True,
    "timestamp_format": "%Y-%m-%d %H:%M:%S"
}

# Error messages and user guidance
ERROR_MESSAGES = {
    "video_load_error": "Could not load video file. Please check the file format and try again.",
    "model_load_error": "Failed to load YOLOv8 model. Please check your internet connection.",
    "processing_error": "An error occurred during video processing. Please try again.",
    "memory_error": "Insufficient memory to process this video. Try a smaller file or lower resolution.",
    "timeout_error": "Processing took too long and was stopped. Try a shorter video or adjust settings."
}

USER_GUIDANCE = {
    "video_upload": "Upload MP4, AVI, MOV, or MKV files up to 500MB. For best results, use videos with clear visibility of people.",
    "threshold_setting": "Lower thresholds = more sensitive alerts. Higher thresholds = fewer false alarms.",
    "grid_configuration": "Smaller grid cells = more detailed analysis. Larger cells = better performance.",
    "venue_selection": "Choose the venue type that best matches your monitoring location for optimal threshold settings."
}

# API and integration settings (for future extensions)
API_SETTINGS = {
    "enable_api": False,
    "api_port": 8000,
    "api_host": "0.0.0.0",
    "rate_limit_requests_per_minute": 60,
    "enable_webhooks": False,
    "webhook_timeout_seconds": 30
}

# Security and privacy settings
SECURITY_SETTINGS = {
    "blur_faces": False,  # Option to blur detected faces for privacy
    "anonymize_data": False,  # Remove personally identifiable information
    "secure_file_handling": True,
    "auto_delete_uploads": True,
    "data_retention_hours": 24
}
