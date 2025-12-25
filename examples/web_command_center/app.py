#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
SAM3 Web Command Center

A Flask-based web interface for real-time object detection and tracking
using SAM3. Features include:
- Live camera feed with segmentation overlay
- Multi-prompt detection configuration
- Object count limits with show/hide functionality
- Claude Vision API integration for detailed object analysis
- Video tracking with memory (SAM3 tracker)
- Multi-object tracking with persistent IDs
- Mask refinement (fill holes, non-overlap)
- Advanced detection controls (boundary/occlusion suppression, hotstart)
- YOLO integration for classification and pose estimation
- Command center style interface with verbose logging

Usage:
    python app.py --prompt "person, car" --camera 0

Then open http://localhost:5000 in your browser.
"""

import argparse
import base64
import io
import ipaddress
import json
import os
import sqlite3
import ssl
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, Response, render_template, request, jsonify
from scipy import ndimage

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    # Look for .env in the web_command_center directory
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
    else:
        # Also check current working directory
        load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system environment

# Add parent directory to path for sam3 imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sam3.utils.device import get_device, get_device_str, setup_device_optimizations, empty_cache

app = Flask(__name__)

# Global API key storage (can be set via CLI arg or environment)
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')


# ===== SAM3 to COCO Label Mapping =====
# Maps open-vocabulary SAM3 labels to COCO class indices for YOLO
SAM3_TO_COCO = {
    # Person variations -> COCO class 0
    "person": 0, "human": 0, "man": 0, "woman": 0, "child": 0, "kid": 0,
    "boy": 0, "girl": 0, "people": 0, "pedestrian": 0, "worker": 0,
    "player": 0, "athlete": 0, "runner": 0, "cyclist": 0,

    # Vehicles
    "bicycle": 1, "bike": 1, "cycle": 1,
    "car": 2, "automobile": 2, "vehicle": 2, "sedan": 2, "suv": 2,
    "motorcycle": 3, "motorbike": 3, "scooter": 3,
    "airplane": 4, "plane": 4, "aircraft": 4, "jet": 4,
    "bus": 5, "coach": 5,
    "train": 6, "locomotive": 6, "railway": 6,
    "truck": 7, "lorry": 7, "pickup": 7,
    "boat": 8, "ship": 8, "vessel": 8, "yacht": 8,

    # Traffic
    "traffic light": 9, "stoplight": 9,
    "fire hydrant": 10, "hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,

    # Animals
    "bird": 14, "sparrow": 14, "pigeon": 14, "crow": 14,
    "cat": 15, "kitten": 15, "feline": 15, "kitty": 15,
    "dog": 16, "puppy": 16, "canine": 16, "hound": 16,
    "horse": 17, "pony": 17, "stallion": 17, "mare": 17,
    "sheep": 18, "lamb": 18,
    "cow": 19, "cattle": 19, "bull": 19,
    "elephant": 20,
    "bear": 21, "grizzly": 21,
    "zebra": 22,
    "giraffe": 23,

    # Accessories
    "backpack": 24, "bag": 24, "rucksack": 24,
    "umbrella": 25, "parasol": 25,
    "handbag": 26, "purse": 26,
    "tie": 27, "necktie": 27,
    "suitcase": 28, "luggage": 28,

    # Sports
    "frisbee": 29,
    "skis": 30, "ski": 30,
    "snowboard": 31,
    "sports ball": 32, "ball": 32, "football": 32, "soccer ball": 32,
    "kite": 33,
    "baseball bat": 34, "bat": 34,
    "baseball glove": 35, "glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38, "racket": 38,

    # Kitchen
    "bottle": 39, "water bottle": 39,
    "wine glass": 40, "glass": 40,
    "cup": 41, "mug": 41, "coffee cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,

    # Food
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52, "hotdog": 52,
    "pizza": 53,
    "donut": 54, "doughnut": 54,
    "cake": 55,

    # Furniture
    "chair": 56, "seat": 56,
    "couch": 57, "sofa": 57,
    "potted plant": 58, "plant": 58, "houseplant": 58,
    "bed": 59,
    "dining table": 60, "table": 60, "desk": 60,
    "toilet": 61,

    # Electronics
    "tv": 62, "television": 62, "monitor": 62, "screen": 62,
    "laptop": 63, "notebook": 63, "computer": 63,
    "mouse": 64, "computer mouse": 64,
    "remote": 65, "remote control": 65,
    "keyboard": 66,
    "cell phone": 67, "phone": 67, "mobile": 67, "smartphone": 67,

    # Appliances
    "microwave": 68,
    "oven": 69, "stove": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72, "fridge": 72,

    # Other
    "book": 73,
    "clock": 74, "watch": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77, "stuffed animal": 77,
    "hair drier": 78, "hairdryer": 78,
    "toothbrush": 79,
}

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Pose keypoint names (COCO format - 17 keypoints)
POSE_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Keypoint colors (BGR)
KEYPOINT_COLORS = {
    'face': (255, 200, 100),      # Light blue for face points
    'left_arm': (0, 255, 0),      # Green for left side
    'right_arm': (0, 0, 255),     # Red for right side
    'left_leg': (0, 200, 0),      # Darker green
    'right_leg': (0, 0, 200),     # Darker red
    'torso': (255, 255, 0),       # Cyan
}


def get_keypoint_color(idx: int) -> Tuple[int, int, int]:
    """Get color for a keypoint based on its index."""
    if idx <= 4:
        return KEYPOINT_COLORS['face']
    elif idx in [5, 7, 9]:
        return KEYPOINT_COLORS['left_arm']
    elif idx in [6, 8, 10]:
        return KEYPOINT_COLORS['right_arm']
    elif idx in [11, 13, 15]:
        return KEYPOINT_COLORS['left_leg']
    elif idx in [12, 14, 16]:
        return KEYPOINT_COLORS['right_leg']
    else:
        return KEYPOINT_COLORS['torso']


# ===== DATABASE =====
class Database:
    """SQLite database for storing all command center data."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'command_center.db')
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table - tracks each app run
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    device TEXT,
                    prompts TEXT,
                    settings TEXT
                )
            ''')

            # Detections table - all detected objects
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    detection_id INTEGER,
                    persistent_id INTEGER,
                    label TEXT,
                    confidence REAL,
                    box TEXT,
                    mask_area INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    frame_number INTEGER,
                    yolo_class TEXT,
                    yolo_confidence REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')

            # Analysis results from Claude
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    detection_id INTEGER,
                    label TEXT,
                    analysis TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_data TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')

            # Location memory - where objects are typically found
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS location_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    context TEXT,
                    position TEXT,
                    frequency INTEGER DEFAULT 1,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(label, context)
                )
            ''')

            # Navigation sessions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS navigation_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    target_label TEXT,
                    target_id INTEGER,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ended_at TIMESTAMP,
                    reached BOOLEAN DEFAULT FALSE,
                    path_history TEXT,
                    scene_context TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')

            # Obstacles detected during navigation
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS obstacles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    navigation_id INTEGER,
                    label TEXT,
                    obstacle_type TEXT,
                    box TEXT,
                    distance TEXT,
                    alert_sent BOOLEAN DEFAULT FALSE,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (navigation_id) REFERENCES navigation_sessions(id)
                )
            ''')

            # Voice queries and results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS voice_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    query TEXT,
                    parsed_prompts TEXT,
                    was_search BOOLEAN,
                    was_describe BOOLEAN,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')

            # General event log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS event_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    event_type TEXT,
                    level TEXT DEFAULT 'INFO',
                    message TEXT,
                    data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            ''')

            # Create indexes for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_session ON detections(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_location_label ON location_memory(label)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_obstacles_nav ON obstacles(navigation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_session ON event_log(session_id)')

            conn.commit()
            print(f"Database initialized: {self.db_path}")

    # ===== SESSION METHODS =====

    def create_session(self, device: str, prompts: List[str], settings: Dict) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        with self.lock:
            with self._get_connection() as conn:
                conn.execute(
                    'INSERT INTO sessions (id, device, prompts, settings) VALUES (?, ?, ?, ?)',
                    (session_id, device, json.dumps(prompts), json.dumps(settings))
                )
                conn.commit()
        return session_id

    def end_session(self, session_id: str):
        """Mark a session as ended."""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute(
                    'UPDATE sessions SET ended_at = CURRENT_TIMESTAMP WHERE id = ?',
                    (session_id,)
                )
                conn.commit()

    # ===== DETECTION METHODS =====

    def save_detection(self, session_id: str, detection: Dict, frame_number: int):
        """Save a detection to the database."""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO detections
                    (session_id, detection_id, persistent_id, label, confidence, box, mask_area, frame_number, yolo_class, yolo_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    detection.get('id'),
                    detection.get('persistent_id'),
                    detection.get('label'),
                    detection.get('confidence'),
                    json.dumps(detection.get('box')),
                    detection.get('mask_area'),
                    frame_number,
                    detection.get('yolo_class'),
                    detection.get('yolo_confidence')
                ))
                conn.commit()

    def save_detections_batch(self, session_id: str, detections: List[Dict], frame_number: int):
        """Save multiple detections in a batch."""
        if not detections:
            return
        with self.lock:
            with self._get_connection() as conn:
                conn.executemany('''
                    INSERT INTO detections
                    (session_id, detection_id, persistent_id, label, confidence, box, mask_area, frame_number, yolo_class, yolo_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [(
                    session_id,
                    d.get('id'),
                    d.get('persistent_id'),
                    d.get('label'),
                    d.get('confidence'),
                    json.dumps(d.get('box')),
                    d.get('mask_area'),
                    frame_number,
                    d.get('yolo_class'),
                    d.get('yolo_confidence')
                ) for d in detections])
                conn.commit()

    def get_detection_history(self, session_id: str = None, label: str = None, limit: int = 100) -> List[Dict]:
        """Get detection history with optional filters."""
        query = 'SELECT * FROM detections WHERE 1=1'
        params = []

        if session_id:
            query += ' AND session_id = ?'
            params.append(session_id)
        if label:
            query += ' AND label LIKE ?'
            params.append(f'%{label}%')

        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ===== ANALYSIS METHODS =====

    def save_analysis(self, session_id: str, detection_id: int, label: str, analysis: str, image_data: str = None):
        """Save Claude analysis result."""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO analysis_results (session_id, detection_id, label, analysis, image_data)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, detection_id, label, analysis, image_data))
                conn.commit()

    def get_analysis_history(self, session_id: str = None, limit: int = 50) -> List[Dict]:
        """Get analysis history."""
        query = 'SELECT * FROM analysis_results'
        params = []

        if session_id:
            query += ' WHERE session_id = ?'
            params.append(session_id)

        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ===== LOCATION MEMORY METHODS =====

    def remember_location(self, label: str, context: str, position: Dict = None):
        """Remember where an object was found."""
        label_key = label.lower().strip()
        context_key = context.lower().strip() if context else ""

        with self.lock:
            with self._get_connection() as conn:
                # Try to update existing entry
                cursor = conn.execute('''
                    UPDATE location_memory
                    SET frequency = frequency + 1,
                        last_seen = CURRENT_TIMESTAMP,
                        position = ?
                    WHERE label = ? AND context = ?
                ''', (json.dumps(position) if position else None, label_key, context_key))

                if cursor.rowcount == 0:
                    # Insert new entry
                    conn.execute('''
                        INSERT INTO location_memory (label, context, position, frequency)
                        VALUES (?, ?, ?, 1)
                    ''', (label_key, context_key, json.dumps(position) if position else None))

                conn.commit()

    def recall_location(self, label: str) -> Optional[Dict]:
        """Recall where an object was typically found."""
        label_key = label.lower().strip()

        with self._get_connection() as conn:
            row = conn.execute('''
                SELECT * FROM location_memory
                WHERE label = ?
                ORDER BY frequency DESC, last_seen DESC
                LIMIT 1
            ''', (label_key,)).fetchone()

            if row:
                result = dict(row)
                if result.get('position'):
                    result['position'] = json.loads(result['position'])
                return result
            return None

    def get_all_location_memories(self) -> List[Dict]:
        """Get all location memories."""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT label, context, frequency, last_seen
                FROM location_memory
                ORDER BY frequency DESC, last_seen DESC
            ''').fetchall()
            return [dict(row) for row in rows]

    def clear_location_memory(self, label: str = None):
        """Clear location memory for a label or all."""
        with self.lock:
            with self._get_connection() as conn:
                if label:
                    conn.execute('DELETE FROM location_memory WHERE label = ?', (label.lower().strip(),))
                else:
                    conn.execute('DELETE FROM location_memory')
                conn.commit()

    # ===== NAVIGATION METHODS =====

    def start_navigation_session(self, session_id: str, target_label: str, target_id: int = None) -> int:
        """Start a new navigation session and return its ID."""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.execute('''
                    INSERT INTO navigation_sessions (session_id, target_label, target_id)
                    VALUES (?, ?, ?)
                ''', (session_id, target_label, target_id))
                conn.commit()
                return cursor.lastrowid

    def end_navigation_session(self, nav_id: int, reached: bool, path_history: List = None, scene_context: Dict = None):
        """End a navigation session."""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute('''
                    UPDATE navigation_sessions
                    SET ended_at = CURRENT_TIMESTAMP,
                        reached = ?,
                        path_history = ?,
                        scene_context = ?
                    WHERE id = ?
                ''', (reached, json.dumps(path_history), json.dumps(scene_context), nav_id))
                conn.commit()

    def save_obstacle(self, nav_id: int, label: str, obstacle_type: str, box: List, distance: str, alert_sent: bool = False):
        """Save an obstacle detected during navigation."""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO obstacles (navigation_id, label, obstacle_type, box, distance, alert_sent)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (nav_id, label, obstacle_type, json.dumps(box), distance, alert_sent))
                conn.commit()

    def get_navigation_history(self, session_id: str = None, limit: int = 20) -> List[Dict]:
        """Get navigation history."""
        query = 'SELECT * FROM navigation_sessions'
        params = []

        if session_id:
            query += ' WHERE session_id = ?'
            params.append(session_id)

        query += ' ORDER BY started_at DESC LIMIT ?'
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ===== VOICE QUERY METHODS =====

    def save_voice_query(self, session_id: str, query: str, parsed_prompts: List[str],
                         was_search: bool = True, was_describe: bool = False):
        """Save a voice query."""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO voice_queries (session_id, query, parsed_prompts, was_search, was_describe)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, query, json.dumps(parsed_prompts), was_search, was_describe))
                conn.commit()

    # ===== EVENT LOG METHODS =====

    def log_event(self, session_id: str, event_type: str, message: str, level: str = 'INFO', data: Dict = None):
        """Log an event to the database."""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute('''
                    INSERT INTO event_log (session_id, event_type, level, message, data)
                    VALUES (?, ?, ?, ?, ?)
                ''', (session_id, event_type, level, message, json.dumps(data) if data else None))
                conn.commit()

    def get_event_log(self, session_id: str = None, event_type: str = None, limit: int = 100) -> List[Dict]:
        """Get event log with optional filters."""
        query = 'SELECT * FROM event_log WHERE 1=1'
        params = []

        if session_id:
            query += ' AND session_id = ?'
            params.append(session_id)
        if event_type:
            query += ' AND event_type = ?'
            params.append(event_type)

        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ===== STATISTICS METHODS =====

    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session."""
        with self._get_connection() as conn:
            stats = {}

            # Detection count
            row = conn.execute(
                'SELECT COUNT(*) as count FROM detections WHERE session_id = ?',
                (session_id,)
            ).fetchone()
            stats['total_detections'] = row['count'] if row else 0

            # Unique labels
            rows = conn.execute(
                'SELECT DISTINCT label FROM detections WHERE session_id = ?',
                (session_id,)
            ).fetchall()
            stats['unique_labels'] = [row['label'] for row in rows]
            stats['unique_label_count'] = len(stats['unique_labels'])

            # Analysis count
            row = conn.execute(
                'SELECT COUNT(*) as count FROM analysis_results WHERE session_id = ?',
                (session_id,)
            ).fetchone()
            stats['total_analyses'] = row['count'] if row else 0

            # Navigation count
            row = conn.execute(
                'SELECT COUNT(*) as count, SUM(CASE WHEN reached THEN 1 ELSE 0 END) as reached FROM navigation_sessions WHERE session_id = ?',
                (session_id,)
            ).fetchone()
            stats['navigation_sessions'] = row['count'] if row else 0
            stats['successful_navigations'] = row['reached'] if row and row['reached'] else 0

            return stats

    def migrate_from_json(self, location_memory_file: str):
        """Migrate existing JSON location memory to SQLite."""
        if not os.path.exists(location_memory_file):
            return

        try:
            with open(location_memory_file, 'r') as f:
                old_memory = json.load(f)

            for label, entries in old_memory.items():
                for entry in entries:
                    self.remember_location(
                        label=label,
                        context=entry.get('context', ''),
                        position=entry.get('position')
                    )
                    # Update frequency if specified
                    if entry.get('frequency', 1) > 1:
                        with self._get_connection() as conn:
                            conn.execute('''
                                UPDATE location_memory
                                SET frequency = ?
                                WHERE label = ? AND context = ?
                            ''', (entry['frequency'], label.lower(), entry.get('context', '').lower()))
                            conn.commit()

            print(f"Migrated {len(old_memory)} items from JSON to SQLite")

            # Optionally rename old file
            backup_path = location_memory_file + '.bak'
            os.rename(location_memory_file, backup_path)
            print(f"Old JSON file backed up to {backup_path}")

        except Exception as e:
            print(f"Error migrating from JSON: {e}")


# Global database instance
db = Database()


# ===== OBSTACLE DEFINITIONS =====
# Common obstacles/hazards for navigation
OBSTACLE_PROMPTS = [
    "stairs", "staircase", "steps",
    "edge", "ledge", "drop", "cliff",
    "door", "doorway", "gate",
    "wall", "pillar", "column", "pole",
    "furniture", "chair", "table", "desk", "couch", "sofa",
    "cable", "wire", "cord",
    "wet floor", "puddle", "spill",
    "hole", "pit", "gap",
    "glass", "window", "mirror",
    "car", "vehicle", "bicycle", "bike",
    "person", "people", "crowd",
    "pet", "dog", "cat", "animal"
]

# Obstacle severity levels
OBSTACLE_SEVERITY = {
    "stairs": "high",
    "staircase": "high",
    "steps": "high",
    "edge": "high",
    "ledge": "high",
    "drop": "high",
    "cliff": "high",
    "hole": "high",
    "pit": "high",
    "gap": "high",
    "wet floor": "medium",
    "puddle": "medium",
    "spill": "medium",
    "cable": "medium",
    "wire": "medium",
    "cord": "medium",
    "car": "high",
    "vehicle": "high",
    "bicycle": "medium",
    "bike": "medium",
    "glass": "medium",
    "door": "low",
    "doorway": "low",
    "wall": "low",
    "pillar": "low",
    "furniture": "low",
    "chair": "low",
    "table": "low",
    "person": "low",
    "people": "medium",
    "crowd": "medium",
}


# Global state
class CommandCenter:
    """Global state manager for the command center."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.paused = False

        # Detection settings
        self.prompts = ["object"]
        self.confidence_threshold = 0.3
        self.max_objects_per_prompt = {}  # prompt -> max count (None = unlimited)
        self.show_all_matches = {}  # prompt -> bool (show all even if over limit)

        # Current detection state
        self.current_detections = []  # List of detection dicts
        self.frame_count = 0
        self.fps = 0.0
        self.device_str = "cpu"

        # Verbose log
        self.log_entries = deque(maxlen=100)

        # Claude analysis results
        self.analysis_queue = []  # Objects waiting for analysis
        self.analysis_results = deque(maxlen=20)  # Recent analysis results
        self.analyzing = False

        # Frame for streaming
        self.current_frame = None  # Frame with overlays (for display)
        self.current_raw_frame = None  # Raw frame without overlays (for analysis)
        self.current_frame_jpeg = None

        # Camera and model
        self.camera = None
        self.processor = None
        self.state = None
        self.video_predictor = None  # SAM3 video predictor for memory tracking

        # Basic tracking state (optical flow)
        self.enable_tracking = True
        self.skip_frames = 3
        self.last_masks = None
        self.last_boxes = None
        self.last_scores = None
        self.last_labels = None
        self.prev_gray = None

        # ===== FEATURE TOGGLES =====

        # Video Tracking with Memory (SAM3 tracker)
        self.enable_memory_tracking = False
        self.memory_bank = {}  # object_id -> list of mask features
        self.memory_max_frames = 10  # Max frames to keep in memory per object

        # Multi-Object Tracking with Persistent IDs
        self.enable_persistent_ids = False
        self.object_registry = {}  # object_id -> {label, first_seen, last_seen, color, ...}
        self.next_object_id = 1
        self.iou_threshold = 0.3  # IoU threshold for matching objects

        # Multi-Object Video Tracking
        self.tracked_objects = {}  # object_id -> tracking state
        self.object_colors = {}  # object_id -> color

        # Mask Refinement Options
        self.enable_fill_holes = False
        self.fill_hole_area = 100  # Max hole area to fill (pixels)
        self.enable_non_overlap = False  # Prevent mask overlaps
        self.enable_smooth_edges = False
        self.smooth_kernel_size = 5

        # Advanced Detection Controls
        self.enable_boundary_suppression = False
        self.boundary_margin = 10  # Pixels from edge to suppress
        self.enable_occlusion_suppression = False
        self.occlusion_threshold = 0.5  # Overlap ratio to suppress
        self.enable_hotstart = False
        self.hotstart_frames = 5  # Frames before confirming new detection
        self.pending_detections = {}  # id -> {frames_seen, detection_data}

        # ===== YOLO FEATURES =====
        self.yolo_classify_model = None
        self.yolo_pose_model = None
        self.yolo_available = False

        # YOLO Classification
        self.enable_yolo_classify = False
        self.yolo_classify_threshold = 0.3
        self.yolo_classify_every_n = 1  # Run classification every N keyframes

        # YOLO Pose Estimation
        self.enable_yolo_pose = False
        self.yolo_pose_threshold = 0.5
        self.show_keypoint_labels = False
        self.show_skeleton = True
        self.keypoint_radius = 4
        self.skeleton_thickness = 2

        # Label spoofing (use SAM3->COCO mapping)
        self.enable_label_spoofing = True

        # Store pose results
        self.last_poses = {}  # object_id -> keypoints

        # ===== VOICE SEARCH =====
        self.voice_enabled = True
        self.last_voice_query = ""
        self.last_parsed_prompts = []
        self.tts_enabled = True
        self.tts_voice = "default"
        self.voice_feedback_messages = deque(maxlen=10)

        # ===== CAMERA SETTINGS =====
        self.current_camera_id = 0
        self.available_cameras = []  # List of {id, name, description}
        self.flip_horizontal = False
        self.flip_vertical = False

        # ===== REFERENCE IMAGE SEARCH =====
        self.clip_model = None
        self.clip_processor = None
        self.clip_available = False
        self.reference_image = None  # PIL Image
        self.reference_embedding = None  # CLIP embedding
        self.reference_description = None  # Text description from Claude
        self.visual_match_threshold = 0.75  # Similarity threshold for CLIP matching
        self.visual_match_enabled = False  # Whether to use CLIP matching

        # ===== GEOMETRIC PROMPTS (Draw to Search) =====
        self.pending_box_prompt = None  # (x1, y1, x2, y2) for box prompt
        self.pending_point_prompt = None  # (x, y) for point prompt
        self.draw_mode = None  # 'box' or 'point'

        # ===== SESSION TRACKING =====
        self.session_id = None  # Current session ID for database

        # ===== NAVIGATION SYSTEM (Accessibility) =====
        self.navigation_active = False
        self.navigation_target = None  # Target object label
        self.navigation_target_id = None  # Target detection ID
        self.navigation_db_id = None  # Navigation session ID in database
        self.navigation_start_time = None
        self.navigation_last_seen = None  # Last position of target
        self.navigation_guidance_queue = deque(maxlen=10)  # Pending guidance messages
        self.navigation_last_guidance = None  # Last spoken guidance
        self.navigation_last_guidance_time = 0
        self.navigation_guidance_interval = 1.5  # Seconds between guidance
        self.navigation_reached = False  # Whether target was reached
        self.navigation_context = None  # Scene context from Claude

        # Navigation spatial tracking
        self.navigation_target_history = []  # History of target positions
        self.navigation_frame_center = (320, 240)  # Frame center (updated dynamically)
        self.navigation_proximity_threshold = 0.25  # Object covers 25% of frame = reachable
        self.navigation_close_threshold = 0.15  # Getting close
        self.navigation_direction_deadzone = 0.1  # Center deadzone

        # ===== OBSTACLE DETECTION =====
        self.obstacle_detection_active = False  # Run obstacle detection during navigation
        self.current_obstacles = []  # Currently detected obstacles
        self.obstacle_alert_cooldown = {}  # obstacle_label -> last_alert_time
        self.obstacle_alert_interval = 3.0  # Seconds between repeated alerts for same obstacle
        self.obstacle_masks = None  # Masks for obstacles to render
        self.obstacle_boxes = None  # Boxes for obstacles

        # ===== LOCATION MEMORY (Now uses SQLite) =====
        self.location_memory_file = os.path.join(os.path.dirname(__file__), '.location_memory.json')
        self._migrate_location_memory()

    def _migrate_location_memory(self):
        """Migrate old JSON location memory to SQLite if it exists."""
        if os.path.exists(self.location_memory_file):
            db.migrate_from_json(self.location_memory_file)

    def remember_location(self, label: str, context: str, position: Dict = None):
        """Remember where an object was found (uses SQLite)."""
        db.remember_location(label, context, position)
        self.log(f"Remembered: {label} found in {context}")

    def recall_location(self, label: str) -> Optional[Dict]:
        """Recall where an object was last found (uses SQLite)."""
        return db.recall_location(label)

    def get_all_location_memories(self) -> List[Dict]:
        """Get all location memories from database."""
        return db.get_all_location_memories()

    def clear_location_memory(self, label: str = None):
        """Clear location memory (uses SQLite)."""
        db.clear_location_memory(label)
        self.log(f"Cleared location memory" + (f" for {label}" if label else ""))

    def _old_recall_location(self, label: str) -> Optional[Dict]:
        """Old recall method - kept for reference."""
        label_key = label.lower().strip()

        if label_key not in self.location_memory:
            return None

        entries = self.location_memory[label_key]
        if not entries:
            return None

        # Return most frequent location, or most recent
        sorted_entries = sorted(entries, key=lambda x: (x.get("frequency", 1), x.get("timestamp", "")), reverse=True)
        return sorted_entries[0]

    def add_navigation_guidance(self, message: str, priority: int = 1):
        """Add a guidance message to the queue."""
        with self.lock:
            self.navigation_guidance_queue.append({
                "message": message,
                "priority": priority,
                "timestamp": time.time()
            })

    def get_pending_guidance(self) -> Optional[str]:
        """Get the next pending guidance message."""
        with self.lock:
            if self.navigation_guidance_queue:
                # Get highest priority message
                sorted_queue = sorted(self.navigation_guidance_queue, key=lambda x: -x["priority"])
                msg = sorted_queue[0]
                self.navigation_guidance_queue.remove(msg)
                return msg["message"]
        return None

    def add_voice_feedback(self, message: str, msg_type: str = "info"):
        """Add a voice feedback message."""
        with self.lock:
            self.voice_feedback_messages.append({
                "message": message,
                "type": msg_type,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

    def log(self, message: str, level: str = "INFO"):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        with self.lock:
            self.log_entries.append(entry)

    def get_logs(self, limit: int = 50) -> List[Dict]:
        """Get recent log entries."""
        with self.lock:
            return list(self.log_entries)[-limit:]

    def add_detection(self, detection: Dict):
        """Add a detection to the current list."""
        with self.lock:
            self.current_detections.append(detection)

    def clear_detections(self):
        """Clear all current detections."""
        with self.lock:
            self.current_detections = []

    def get_filtered_detections(self) -> Tuple[List[Dict], Dict]:
        """Get detections filtered by max count settings."""
        with self.lock:
            detections = self.current_detections.copy()

        # Group by prompt
        by_prompt = {}
        for det in detections:
            prompt = det.get("label", "unknown")
            if prompt not in by_prompt:
                by_prompt[prompt] = []
            by_prompt[prompt].append(det)

        # Apply filters
        filtered = []
        hidden_counts = {}

        for prompt, dets in by_prompt.items():
            max_count = self.max_objects_per_prompt.get(prompt)
            show_all = self.show_all_matches.get(prompt, False)

            if max_count is not None and not show_all:
                dets_sorted = sorted(dets, key=lambda d: d.get("confidence", 0), reverse=True)
                filtered.extend(dets_sorted[:max_count])
                hidden = len(dets_sorted) - max_count
                if hidden > 0:
                    hidden_counts[prompt] = hidden
            else:
                filtered.extend(dets)

        return filtered, hidden_counts

    def queue_analysis(self, detection_id: int, image_data: str):
        """Queue an object for Claude analysis."""
        with self.lock:
            self.analysis_queue.append({
                "id": detection_id,
                "image_data": image_data,
                "timestamp": datetime.now().isoformat()
            })

    def add_analysis_result(self, detection_id: int, result: str):
        """Add a Claude analysis result."""
        with self.lock:
            self.analysis_results.append({
                "id": detection_id,
                "result": result,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

    def get_feature_status(self) -> Dict:
        """Get status of all feature toggles."""
        return {
            "tracking": self.enable_tracking,
            "memory_tracking": self.enable_memory_tracking,
            "persistent_ids": self.enable_persistent_ids,
            "fill_holes": self.enable_fill_holes,
            "non_overlap": self.enable_non_overlap,
            "smooth_edges": self.enable_smooth_edges,
            "boundary_suppression": self.enable_boundary_suppression,
            "occlusion_suppression": self.enable_occlusion_suppression,
            "hotstart": self.enable_hotstart,
            "yolo_classify": self.enable_yolo_classify,
            "yolo_pose": self.enable_yolo_pose,
            "show_keypoint_labels": self.show_keypoint_labels,
            "show_skeleton": self.show_skeleton,
            "label_spoofing": self.enable_label_spoofing,
        }


# Global command center instance
cc = CommandCenter()


# Color palette (BGR for OpenCV)
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 255),  # Purple
    (255, 128, 0),  # Orange
    (128, 255, 0),  # Lime
    (0, 128, 255),  # Sky blue
]


def load_yolo_models():
    """Load YOLO models for classification and pose estimation."""
    global cc

    try:
        from ultralytics import YOLO

        cc.log("Loading YOLO models...")

        # Model priority: YOLO12 -> YOLO11 -> YOLOv8
        # YOLO12 is newest (Feb 2025) but pretrained weights may not be available for all tasks
        cls_models = ['yolo12n-cls.pt', 'yolo11n-cls.pt', 'yolov8n-cls.pt']
        pose_models = ['yolo12n-pose.pt', 'yolo11n-pose.pt', 'yolov8n-pose.pt']

        # Load classification model
        cc.yolo_classify_model = None
        for model_name in cls_models:
            try:
                cc.yolo_classify_model = YOLO(model_name)
                cc.log(f"YOLO classification model loaded ({model_name})", "SUCCESS")
                break
            except Exception as e:
                cc.log(f"Could not load {model_name}: {e}", "WARN")
                continue

        if cc.yolo_classify_model is None:
            cc.log("No classification model available", "WARN")

        # Load pose estimation model
        cc.yolo_pose_model = None
        for model_name in pose_models:
            try:
                cc.yolo_pose_model = YOLO(model_name)
                cc.log(f"YOLO pose model loaded ({model_name})", "SUCCESS")
                break
            except Exception as e:
                cc.log(f"Could not load {model_name}: {e}", "WARN")
                continue

        if cc.yolo_pose_model is None:
            cc.log("No pose model available", "WARN")

        cc.yolo_available = cc.yolo_classify_model is not None or cc.yolo_pose_model is not None

        if cc.yolo_available:
            cc.log("YOLO models ready", "SUCCESS")
        else:
            cc.log("No YOLO models available", "WARN")

    except ImportError:
        cc.log("ultralytics not installed. YOLO features disabled. Install with: pip install ultralytics", "WARN")
        cc.yolo_available = False


def load_clip_model():
    """Load CLIP model for visual similarity matching."""
    global cc

    try:
        from transformers import CLIPProcessor, CLIPModel

        cc.log("Loading CLIP model for visual matching...")

        # Use a smaller/faster CLIP model
        model_name = "openai/clip-vit-base-patch32"

        cc.clip_processor = CLIPProcessor.from_pretrained(model_name)
        cc.clip_model = CLIPModel.from_pretrained(model_name)

        # Move to appropriate device
        device = get_device()
        cc.clip_model = cc.clip_model.to(device)
        cc.clip_model.eval()

        cc.clip_available = True
        cc.log("CLIP model loaded successfully", "SUCCESS")

    except ImportError:
        cc.log("transformers not installed. Visual matching disabled. Install with: pip install transformers", "WARN")
        cc.clip_available = False
    except Exception as e:
        cc.log(f"Failed to load CLIP model: {e}", "ERROR")
        cc.clip_available = False


def get_clip_embedding(image: Image.Image) -> Optional[torch.Tensor]:
    """Get CLIP embedding for an image."""
    global cc

    if not cc.clip_available or cc.clip_model is None:
        return None

    try:
        device = get_device()
        inputs = cc.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            embedding = cc.clip_model.get_image_features(**inputs)
            # Normalize
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding

    except Exception as e:
        cc.log(f"Failed to get CLIP embedding: {e}", "ERROR")
        return None


def compute_clip_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """Compute cosine similarity between two CLIP embeddings."""
    if embedding1 is None or embedding2 is None:
        return 0.0

    with torch.no_grad():
        similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
        return float(similarity.item())


def describe_image_with_claude(image_data: str) -> Optional[str]:
    """Use Claude to generate a detailed description of an image for search."""
    global ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this object concisely for visual detection. Focus on: type of object, color, distinctive features, shape. Return ONLY the description phrase (e.g., 'red baseball cap with white Nike logo', 'black leather handbag with gold clasp'). No other text."
                        }
                    ],
                }
            ],
        )

        return message.content[0].text.strip()

    except Exception as e:
        cc.log(f"Failed to describe image with Claude: {e}", "ERROR")
        return None


# ===== NAVIGATION SYSTEM FUNCTIONS =====

def analyze_scene_context(image_data: str) -> Optional[Dict]:
    """
    Use Claude to analyze the scene for navigation context.
    Returns location type, obstacles, and spatial awareness info.
    """
    global ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Analyze this scene for navigation assistance. Return JSON only:
{
    "location": "room type (kitchen, living room, bedroom, bathroom, office, hallway, outdoor, etc.)",
    "obstacles": ["list of obstacles or hazards visible"],
    "surfaces": ["tables, counters, shelves visible"],
    "lighting": "bright/dim/dark",
    "space": "open/cluttered/narrow",
    "landmarks": ["notable items that help orient"]
}"""
                        }
                    ],
                }
            ],
        )

        response_text = message.content[0].text.strip()

        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        return json.loads(response_text)

    except Exception as e:
        cc.log(f"Scene analysis failed: {e}", "WARN")
        return None


def compute_navigation_guidance(target_box: List[float], frame_shape: Tuple[int, int]) -> Dict:
    """
    Compute navigation guidance based on target position in frame.

    Returns:
        direction: 'left', 'right', 'center', 'up', 'down'
        distance: 'far', 'medium', 'close', 'reachable'
        guidance_text: Human-readable guidance
        arrow_angle: Angle for AR arrow (degrees)
        confidence: How confident we are in the guidance
    """
    global cc

    if not target_box:
        return {
            "direction": "unknown",
            "distance": "unknown",
            "guidance_text": "Looking for target...",
            "arrow_angle": 0,
            "confidence": 0
        }

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = target_box

    # Object center
    obj_center_x = (x1 + x2) / 2
    obj_center_y = (y1 + y2) / 2

    # Frame center
    frame_center_x = w / 2
    frame_center_y = h / 2

    # Normalized position (-1 to 1, 0 = center)
    norm_x = (obj_center_x - frame_center_x) / (w / 2)
    norm_y = (obj_center_y - frame_center_y) / (h / 2)

    # Object size relative to frame
    obj_width = x2 - x1
    obj_height = y2 - y1
    obj_area_ratio = (obj_width * obj_height) / (w * h)

    # Determine direction
    deadzone = cc.navigation_direction_deadzone

    if abs(norm_x) < deadzone and abs(norm_y) < deadzone:
        direction = "center"
        h_dir = ""
    elif abs(norm_x) > abs(norm_y):
        direction = "right" if norm_x > 0 else "left"
        h_dir = direction
    else:
        direction = "down" if norm_y > 0 else "up"
        h_dir = ""

    # Secondary direction
    if direction in ["center"]:
        secondary = ""
    elif direction in ["left", "right"]:
        if norm_y < -deadzone:
            secondary = " and up"
        elif norm_y > deadzone:
            secondary = " and down"
        else:
            secondary = ""
    else:
        if norm_x < -deadzone:
            secondary = " and left"
        elif norm_x > deadzone:
            secondary = " and right"
        else:
            secondary = ""

    # Determine distance based on object size
    if obj_area_ratio >= cc.navigation_proximity_threshold:
        distance = "reachable"
    elif obj_area_ratio >= cc.navigation_close_threshold:
        distance = "close"
    elif obj_area_ratio >= 0.05:
        distance = "medium"
    else:
        distance = "far"

    # Calculate arrow angle (0 = up, 90 = right, etc.)
    import math
    arrow_angle = math.degrees(math.atan2(norm_x, -norm_y))

    # Generate guidance text
    if distance == "reachable":
        if direction == "center":
            guidance_text = "Object is directly in front of you, within reach!"
        else:
            guidance_text = f"Object is within reach, slightly to the {direction}{secondary}"
    elif distance == "close":
        if direction == "center":
            guidance_text = "Almost there! Object is straight ahead, getting close"
        else:
            guidance_text = f"Getting close! Turn {direction}{secondary}"
    elif distance == "medium":
        if direction == "center":
            guidance_text = "Keep moving forward, object ahead"
        else:
            guidance_text = f"Object is to the {direction}{secondary}, move that way"
    else:  # far
        if direction == "center":
            guidance_text = "Object detected ahead, continue forward"
        else:
            guidance_text = f"Object is far to the {direction}{secondary}"

    return {
        "direction": direction,
        "secondary": secondary.strip(),
        "distance": distance,
        "guidance_text": guidance_text,
        "arrow_angle": arrow_angle,
        "norm_x": norm_x,
        "norm_y": norm_y,
        "obj_area_ratio": obj_area_ratio,
        "confidence": min(1.0, obj_area_ratio * 10 + 0.5)  # Higher for larger objects
    }


def get_navigation_status() -> Dict:
    """Get current navigation status and guidance."""
    global cc

    if not cc.navigation_active:
        return {
            "active": False,
            "target": None,
            "guidance": None
        }

    # Find target in current detections
    target_detection = None
    for det in cc.current_detections:
        if det.get("label", "").lower() == cc.navigation_target.lower():
            target_detection = det
            break
        if cc.navigation_target_id is not None and det.get("id") == cc.navigation_target_id:
            target_detection = det
            break

    if target_detection:
        cc.navigation_last_seen = target_detection
        box = target_detection.get("box")

        if cc.current_raw_frame is not None:
            frame_shape = cc.current_raw_frame.shape
        else:
            frame_shape = (480, 640)

        guidance = compute_navigation_guidance(box, frame_shape)

        # Check if reached
        if guidance["distance"] == "reachable" and not cc.navigation_reached:
            cc.navigation_reached = True
            guidance["reached"] = True
            guidance["guidance_text"] = f"You've reached the {cc.navigation_target}! It's right in front of you."

        return {
            "active": True,
            "target": cc.navigation_target,
            "target_visible": True,
            "guidance": guidance,
            "reached": cc.navigation_reached,
            "context": cc.navigation_context,
            "duration": time.time() - cc.navigation_start_time if cc.navigation_start_time else 0
        }
    else:
        # Target not currently visible
        last_guidance = None
        if cc.navigation_last_seen:
            box = cc.navigation_last_seen.get("box")
            if box:
                frame_shape = (480, 640)
                if cc.current_raw_frame is not None:
                    frame_shape = cc.current_raw_frame.shape
                last_guidance = compute_navigation_guidance(box, frame_shape)
                last_guidance["guidance_text"] = f"Lost sight of {cc.navigation_target}. Last seen to the {last_guidance['direction']}"

        return {
            "active": True,
            "target": cc.navigation_target,
            "target_visible": False,
            "guidance": last_guidance or {
                "direction": "unknown",
                "distance": "unknown",
                "guidance_text": f"Looking for {cc.navigation_target}... Turn slowly to scan the area",
                "arrow_angle": 0
            },
            "reached": False,
            "context": cc.navigation_context,
            "searching": True
        }


def get_coco_class_for_label(sam3_label: str) -> Optional[int]:
    """Get COCO class ID for a SAM3 label using the mapping."""
    label_lower = sam3_label.lower().strip()

    # Direct lookup
    if label_lower in SAM3_TO_COCO:
        return SAM3_TO_COCO[label_lower]

    # Try partial match
    for key, coco_id in SAM3_TO_COCO.items():
        if key in label_lower or label_lower in key:
            return coco_id

    return None


def is_person_label(label: str) -> bool:
    """Check if a label refers to a person."""
    coco_id = get_coco_class_for_label(label)
    return coco_id == 0


def classify_region(frame: np.ndarray, box: List[float], sam3_label: str) -> Dict:
    """
    Run YOLO classification on a detected region.

    Returns dict with:
        - yolo_class: Top predicted class name
        - yolo_confidence: Confidence score
        - top5_classes: List of top 5 (class, confidence) tuples
        - matches_sam3: Whether YOLO agrees with SAM3 label
    """
    if cc.yolo_classify_model is None:
        return None

    try:
        # Crop region from frame
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]

        # Add padding
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Run classification
        results = cc.yolo_classify_model(crop, verbose=False)

        if len(results) == 0:
            return None

        probs = results[0].probs

        if probs is None:
            return None

        # Get top 5 predictions
        top5_indices = probs.top5
        top5_confs = probs.top5conf.cpu().numpy()

        # Get class names from model
        names = cc.yolo_classify_model.names

        top5_classes = [(names[idx], float(conf)) for idx, conf in zip(top5_indices, top5_confs)]

        top_class = top5_classes[0][0] if top5_classes else "unknown"
        top_conf = top5_classes[0][1] if top5_classes else 0.0

        # Check if YOLO agrees with SAM3
        sam3_coco_id = get_coco_class_for_label(sam3_label)
        matches = False

        if sam3_coco_id is not None and sam3_coco_id < len(COCO_CLASSES):
            sam3_coco_name = COCO_CLASSES[sam3_coco_id]
            # Check if any top-5 class matches
            for cls_name, conf in top5_classes:
                if cls_name.lower() == sam3_coco_name.lower() or sam3_coco_name.lower() in cls_name.lower():
                    matches = True
                    break

        return {
            "yolo_class": top_class,
            "yolo_confidence": top_conf,
            "top5_classes": top5_classes,
            "matches_sam3": matches
        }

    except Exception as e:
        cc.log(f"YOLO classification error: {e}", "ERROR")
        return None


def estimate_pose(frame: np.ndarray, box: List[float]) -> Dict:
    """
    Run YOLO pose estimation on a person region.

    Returns dict with:
        - keypoints: List of 17 (x, y, confidence) tuples
        - keypoint_names: List of keypoint names
        - confidence: Overall pose confidence
    """
    if cc.yolo_pose_model is None:
        return None

    try:
        # Crop region from frame (with extra padding for pose)
        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = frame.shape[:2]

        # Add generous padding for pose estimation
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * 0.2)
        pad_y = int(box_h * 0.1)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Run pose estimation
        results = cc.yolo_pose_model(crop, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            return None

        keypoints_data = results[0].keypoints

        if keypoints_data.xy is None or len(keypoints_data.xy) == 0:
            return None

        # Get first person's keypoints (we're analyzing one box at a time)
        kpts = keypoints_data.xy[0].cpu().numpy()  # Shape: (17, 2)
        confs = keypoints_data.conf[0].cpu().numpy() if keypoints_data.conf is not None else np.ones(17)

        # Convert coordinates back to full frame
        keypoints = []
        for i, (pt, conf) in enumerate(zip(kpts, confs)):
            # Add offset back to get coordinates in original frame
            frame_x = pt[0] + x1
            frame_y = pt[1] + y1
            keypoints.append((float(frame_x), float(frame_y), float(conf)))

        # Calculate overall confidence
        valid_confs = [c for x, y, c in keypoints if c > 0.1]
        overall_conf = np.mean(valid_confs) if valid_confs else 0.0

        return {
            "keypoints": keypoints,
            "keypoint_names": POSE_KEYPOINTS,
            "confidence": float(overall_conf),
            "box_offset": (x1, y1)  # For reference
        }

    except Exception as e:
        cc.log(f"YOLO pose estimation error: {e}", "ERROR")
        return None


def draw_pose_overlay(frame: np.ndarray, pose_data: Dict, object_id: int = None) -> np.ndarray:
    """Draw pose keypoints and skeleton on frame."""
    if pose_data is None or "keypoints" not in pose_data:
        return frame

    overlay = frame.copy()
    keypoints = pose_data["keypoints"]

    # Draw skeleton connections first (so points are on top)
    if cc.show_skeleton:
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                x1, y1, c1 = keypoints[start_idx]
                x2, y2, c2 = keypoints[end_idx]

                # Only draw if both points have sufficient confidence
                if c1 > cc.yolo_pose_threshold and c2 > cc.yolo_pose_threshold:
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))

                    # Get color based on connection type
                    color = get_keypoint_color(start_idx)
                    cv2.line(overlay, pt1, pt2, color, cc.skeleton_thickness)

    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > cc.yolo_pose_threshold:
            pt = (int(x), int(y))
            color = get_keypoint_color(i)

            # Draw filled circle
            cv2.circle(overlay, pt, cc.keypoint_radius, color, -1)
            # Draw outline
            cv2.circle(overlay, pt, cc.keypoint_radius, (255, 255, 255), 1)

            # Draw label if enabled
            if cc.show_keypoint_labels and i < len(POSE_KEYPOINTS):
                label = POSE_KEYPOINTS[i].replace('_', ' ')
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)

                # Position label above point
                label_x = int(x - tw/2)
                label_y = int(y - cc.keypoint_radius - 3)

                # Background
                cv2.rectangle(overlay,
                    (label_x - 1, label_y - th - 1),
                    (label_x + tw + 1, label_y + 1),
                    (0, 0, 0), -1)

                # Text
                cv2.putText(overlay, label, (label_x, label_y),
                    font, font_scale, (255, 255, 255), 1)

    return overlay


def load_model(checkpoint_path: Optional[str] = None):
    """Load the SAM3 model."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    cc.log("Loading SAM3 model...")
    cc.device_str = get_device_str()

    # Setup device-specific optimizations (MPS memory, CUDA TF32, etc.)
    setup_device_optimizations()
    cc.log(f"Device optimizations enabled for {cc.device_str}")

    model = build_sam3_image_model(
        device=cc.device_str,
        checkpoint_path=checkpoint_path,
        load_from_HF=checkpoint_path is None,
        eval_mode=True,
        enable_segmentation=True,
    )

    cc.processor = Sam3Processor(
        model=model,
        resolution=1008,
        device=cc.device_str,
        confidence_threshold=cc.confidence_threshold,
    )

    cc.log(f"Model loaded on {cc.device_str}", "SUCCESS")

    # Load YOLO models
    load_yolo_models()

    # Load CLIP model for visual matching (optional)
    load_clip_model()


# ===== CAMERA FUNCTIONS =====

def detect_available_cameras(max_cameras: int = 10) -> List[Dict]:
    """
    Detect available cameras on the system.

    Returns list of dicts with:
        - id: Camera index
        - name: Camera name/description
        - resolution: (width, height) if detectable
    """
    cameras = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Try to get backend name
            backend = cap.getBackendName()

            # Create descriptive name
            if i == 0:
                name = "Default Camera"
            else:
                name = f"Camera {i}"

            # Add platform-specific hints
            import platform
            if platform.system() == "Darwin":  # macOS
                if i == 0:
                    name = "FaceTime HD Camera (Built-in)"
                elif i == 1:
                    name = "External Camera"
            elif platform.system() == "Linux":
                # Try to read device name from v4l2
                try:
                    import subprocess
                    result = subprocess.run(
                        ['v4l2-ctl', '--device', f'/dev/video{i}', '--info'],
                        capture_output=True, text=True, timeout=1
                    )
                    for line in result.stdout.split('\n'):
                        if 'Card type' in line:
                            name = line.split(':')[1].strip()
                            break
                except Exception:
                    pass

            cameras.append({
                "id": i,
                "name": name,
                "resolution": f"{width}x{height}",
                "fps": fps,
                "backend": backend,
                "description": f"{name} ({width}x{height} @ {fps:.0f}fps)"
            })

            cap.release()

    return cameras


def switch_camera(camera_id: int) -> bool:
    """Switch to a different camera and reset detection state."""
    global cc

    cc.log(f"Switching to camera {camera_id}...")

    # Release current camera
    if cc.camera is not None:
        cc.camera.release()
        cc.camera = None

    # Open new camera
    new_camera = cv2.VideoCapture(camera_id)

    if not new_camera.isOpened():
        cc.log(f"Failed to open camera {camera_id}", "ERROR")
        # Try to reopen previous camera
        cc.camera = cv2.VideoCapture(cc.current_camera_id)
        return False

    cc.camera = new_camera
    cc.current_camera_id = camera_id

    # Get camera info
    width = int(cc.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cc.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Reset detection state
    reset_detection_state()

    cc.log(f"Switched to camera {camera_id} ({width}x{height})", "SUCCESS")
    return True


def reset_detection_state():
    """Reset all detection state for a fresh start."""
    global cc

    cc.state = None
    cc.last_masks = None
    cc.last_boxes = None
    cc.last_scores = None
    cc.last_labels = None
    cc.tracked_objects = {}
    cc.memory_bank = {}
    cc.object_colors = {}
    cc.next_object_id = 1
    cc.pending_detections = {}
    cc.last_poses = {}
    cc.prev_gray = None
    cc.current_detections = []
    cc.frame_count = 0


# ===== MASK REFINEMENT FUNCTIONS =====

def fill_holes_in_mask(mask: np.ndarray, max_hole_area: int = 100) -> np.ndarray:
    """Fill small holes in a binary mask."""
    mask_bool = mask.astype(bool)
    # Find holes (inverted connected components)
    inverted = ~mask_bool
    labeled, num_features = ndimage.label(inverted)

    # Fill small holes
    for i in range(1, num_features + 1):
        hole = labeled == i
        if hole.sum() <= max_hole_area:
            mask_bool[hole] = True

    return mask_bool.astype(np.float32)


def smooth_mask_edges(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooth mask edges using morphological operations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Close then open to smooth
    smoothed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    return smoothed.astype(np.float32)


def remove_mask_overlaps(masks: List[np.ndarray], scores: List[float]) -> List[np.ndarray]:
    """Remove overlapping regions, keeping higher confidence masks."""
    if len(masks) <= 1:
        return masks

    # Sort by score (highest first)
    sorted_indices = np.argsort(scores)[::-1]
    result_masks = [None] * len(masks)
    occupied = np.zeros_like(masks[0], dtype=bool)

    for idx in sorted_indices:
        mask = masks[idx].astype(bool)
        # Remove already occupied regions
        mask = mask & ~occupied
        result_masks[idx] = mask.astype(np.float32)
        occupied |= mask

    return result_masks


# ===== DETECTION CONTROL FUNCTIONS =====

def is_near_boundary(box: List[float], frame_shape: Tuple[int, int], margin: int = 10) -> bool:
    """Check if a bounding box is near the frame boundary."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    return x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate Intersection over Union between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def match_detection_to_object(mask: np.ndarray, existing_masks: Dict[int, np.ndarray],
                               threshold: float = 0.3) -> Optional[int]:
    """Match a detection to an existing tracked object by IoU."""
    best_match = None
    best_iou = threshold

    for obj_id, existing_mask in existing_masks.items():
        iou = calculate_iou(mask, existing_mask)
        if iou > best_iou:
            best_iou = iou
            best_match = obj_id

    return best_match


def get_bounding_box_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    """Extract bounding box from a binary mask."""
    if mask is None or mask.sum() == 0:
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return [float(x_min), float(y_min), float(x_max), float(y_max)]


def is_mask_valid(mask: np.ndarray, frame_shape: Tuple[int, int], min_area: int = 50,
                   boundary_margin: int = 5) -> bool:
    """
    Check if a tracked mask is still valid.
    Returns False if:
    - Mask is too small (object left frame)
    - Mask is mostly outside the frame boundaries
    """
    if mask is None:
        return False

    mask_area = mask.sum()
    if mask_area < min_area:
        return False

    h, w = frame_shape[:2]

    # Check if mask is mostly within frame bounds
    if mask.shape != (h, w):
        return False

    # Get bounding box
    box = get_bounding_box_from_mask(mask)
    if box is None:
        return False

    x1, y1, x2, y2 = box

    # Check if box is mostly outside frame
    if x2 < boundary_margin or x1 > w - boundary_margin:
        return False
    if y2 < boundary_margin or y1 > h - boundary_margin:
        return False

    return True


def update_detections_from_tracked_masks(tracked_masks: torch.Tensor, frame_shape: Tuple[int, int]):
    """
    Update current_detections based on tracked masks.
    Removes detections for masks that are no longer valid (left frame).
    Updates bounding boxes for masks that moved.
    """
    global cc

    if tracked_masks is None or len(cc.current_detections) == 0:
        return

    h, w = frame_shape[:2]
    masks_np = tracked_masks.squeeze(1).cpu().numpy()

    updated_detections = []
    valid_mask_indices = []

    for i, det in enumerate(cc.current_detections):
        if i >= len(masks_np):
            break

        mask = masks_np[i]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5

        # Check if mask is still valid
        if is_mask_valid(mask, frame_shape):
            # Update bounding box from tracked mask
            new_box = get_bounding_box_from_mask(mask)
            if new_box:
                det = det.copy()  # Don't modify original
                det["box"] = new_box
                det["tracked"] = True  # Mark as being tracked (not fresh detection)
            updated_detections.append(det)
            valid_mask_indices.append(i)
        else:
            # Object has left the frame or tracking failed
            label = det.get("label", "object")
            obj_id = det.get("id", i)
            cc.log(f"Object #{obj_id} ({label}) left frame", "INFO")

    # Update global state
    with cc.lock:
        cc.current_detections = updated_detections

    return valid_mask_indices


# ===== MEMORY TRACKING FUNCTIONS =====

def update_memory_bank(object_id: int, mask_features: torch.Tensor):
    """Update memory bank for an object."""
    if object_id not in cc.memory_bank:
        cc.memory_bank[object_id] = []

    cc.memory_bank[object_id].append(mask_features)

    # Keep only recent frames
    if len(cc.memory_bank[object_id]) > cc.memory_max_frames:
        cc.memory_bank[object_id].pop(0)


# ===== OBSTACLE DETECTION =====

def detect_obstacles(frame: np.ndarray, pil_image: Image.Image) -> List[Dict]:
    """Detect obstacles in the current frame during navigation."""
    global cc

    if not cc.obstacle_detection_active or cc.processor is None:
        return []

    obstacles = []
    current_time = time.time()

    # Create a temporary state for obstacle detection
    try:
        obstacle_state = cc.processor.set_image(pil_image, {})

        # Try to detect common obstacles
        for obstacle_prompt in OBSTACLE_PROMPTS[:10]:  # Limit to top 10 for performance
            # Skip if this is our target
            if cc.navigation_target and obstacle_prompt.lower() in cc.navigation_target.lower():
                continue

            obstacle_state = cc.processor.set_text_prompt(obstacle_prompt, obstacle_state)

            masks = obstacle_state.get("masks")
            boxes = obstacle_state.get("boxes")
            scores = obstacle_state.get("scores")

            if masks is not None and masks.numel() > 0:
                for i in range(min(len(masks), 3)):  # Max 3 per type
                    score = float(scores[i].cpu()) if scores is not None and i < len(scores) else 0.0

                    if score < 0.4:  # Higher threshold for obstacles
                        continue

                    mask_np = masks[i].squeeze().cpu().numpy()
                    box = boxes[i].cpu().numpy().tolist() if boxes is not None and i < len(boxes) else None

                    if box is None:
                        continue

                    # Calculate distance based on box position/size in frame
                    h, w = frame.shape[:2]
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    frame_area = w * h
                    area_ratio = box_area / frame_area

                    # Determine distance
                    if area_ratio > 0.25:
                        distance = "very_close"
                    elif area_ratio > 0.10:
                        distance = "close"
                    elif area_ratio > 0.05:
                        distance = "medium"
                    else:
                        distance = "far"

                    # Get severity
                    severity = OBSTACLE_SEVERITY.get(obstacle_prompt, "low")

                    obstacle = {
                        "label": obstacle_prompt,
                        "type": severity,
                        "box": box,
                        "mask": mask_np,
                        "confidence": score,
                        "distance": distance,
                        "timestamp": current_time
                    }

                    # Check cooldown for alerts
                    cooldown_key = f"{obstacle_prompt}_{distance}"
                    last_alert = cc.obstacle_alert_cooldown.get(cooldown_key, 0)

                    if current_time - last_alert > cc.obstacle_alert_interval:
                        obstacle["should_alert"] = True
                        cc.obstacle_alert_cooldown[cooldown_key] = current_time
                    else:
                        obstacle["should_alert"] = False

                    obstacles.append(obstacle)

                    # Save to database
                    if cc.navigation_db_id and obstacle["should_alert"]:
                        db.save_obstacle(
                            cc.navigation_db_id,
                            obstacle_prompt,
                            severity,
                            box,
                            distance,
                            alert_sent=True
                        )

    except Exception as e:
        cc.log(f"Obstacle detection error: {e}", "ERROR")

    return obstacles


def overlay_obstacles(display: np.ndarray, obstacles: List[Dict]) -> np.ndarray:
    """Overlay obstacle masks and alerts on the display frame."""
    if not obstacles:
        return display

    # Obstacle color (orange/red based on severity)
    colors = {
        "high": (0, 0, 255),      # Red
        "medium": (0, 165, 255),   # Orange
        "low": (0, 255, 255)       # Yellow
    }

    for obstacle in obstacles:
        mask = obstacle.get("mask")
        box = obstacle.get("box")
        severity = obstacle.get("type", "low")
        label = obstacle.get("label", "Obstacle")
        distance = obstacle.get("distance", "unknown")

        color = colors.get(severity, (0, 255, 255))

        # Draw mask overlay
        if mask is not None:
            mask_bool = mask.astype(bool)
            # Create colored overlay
            overlay = display.copy()
            overlay[mask_bool] = color
            # Blend with original (more transparent than regular detections)
            alpha = 0.4 if severity == "high" else 0.3
            display = cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0)

            # Draw mask outline
            contours, _ = cv2.findContours(
                mask.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(display, contours, -1, color, 2)

        # Draw bounding box
        if box:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # Draw alert icon (warning triangle)
            icon_size = 30
            icon_x = x1 + 5
            icon_y = y1 - icon_size - 5 if y1 > icon_size + 10 else y1 + 5

            # Draw warning triangle
            triangle = np.array([
                [icon_x + icon_size // 2, icon_y],
                [icon_x, icon_y + icon_size],
                [icon_x + icon_size, icon_y + icon_size]
            ], np.int32)
            cv2.fillPoly(display, [triangle], color)
            cv2.polylines(display, [triangle], True, (0, 0, 0), 2)

            # Draw exclamation mark
            cv2.line(display, (icon_x + icon_size // 2, icon_y + 8),
                     (icon_x + icon_size // 2, icon_y + icon_size - 12), (0, 0, 0), 2)
            cv2.circle(display, (icon_x + icon_size // 2, icon_y + icon_size - 6), 2, (0, 0, 0), -1)

            # Draw label
            label_text = f"OBSTACLE: {label}"
            if distance in ["very_close", "close"]:
                label_text = f"WARNING: {label} ({distance})"

            text_y = y1 - icon_size - 10 if y1 > icon_size + 30 else y2 + 20
            cv2.putText(display, label_text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(display, label_text, (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return display


# ===== FRAME PROCESSING =====

def process_frame(frame: np.ndarray) -> np.ndarray:
    """Process a frame through SAM3 and overlay results."""
    global cc

    cc.frame_count += 1
    is_keyframe = cc.frame_count % cc.skip_frames == 0

    # Handle geometric prompts (draw to search)
    if cc.pending_box_prompt is not None or cc.pending_point_prompt is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        cc.state = cc.processor.set_image(pil_image, cc.state)

        if cc.pending_box_prompt is not None:
            # Box prompt
            x1, y1, x2, y2 = cc.pending_box_prompt
            cc.state["geometric_prompt"] = {
                "type": "box",
                "box": [x1, y1, x2, y2]
            }
            cc.log(f"Processing box prompt: ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})")

        elif cc.pending_point_prompt is not None:
            # Point prompt
            x, y = cc.pending_point_prompt
            cc.state["geometric_prompt"] = {
                "type": "point",
                "point": [x, y],
                "label": 1  # 1 = foreground, 0 = background
            }
            cc.log(f"Processing point prompt: ({x:.0f},{y:.0f})")

        # Get mask from geometric prompt
        try:
            # Use the processor's segment method with geometric prompt
            masks = cc.state.get("masks")
            if masks is not None and len(masks) > 0:
                mask_np = masks[0].squeeze().cpu().numpy()
                box = get_bounding_box_from_mask(mask_np)

                cc.last_masks = masks[:1]
                cc.last_boxes = torch.tensor([box]) if box else None
                cc.last_scores = torch.tensor([1.0])
                cc.last_labels = ["selected object"]

                # Add to detections
                with cc.lock:
                    cc.current_detections = [{
                        "id": 0,
                        "label": "selected object",
                        "confidence": 1.0,
                        "box": box,
                        "tracked": False,
                    }]

                cc.log("Object selected via drawing", "SUCCESS")

        except Exception as e:
            cc.log(f"Geometric prompt failed: {e}", "ERROR")

        # Clear the pending prompts
        cc.pending_box_prompt = None
        cc.pending_point_prompt = None
        cc.draw_mode = None
        cc.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    elif is_keyframe and not cc.paused:
        # Full inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        cc.state = cc.processor.set_image(pil_image, cc.state)

        # Build new detections list (don't clear until we have new ones)
        new_detections = []
        cc.last_poses = {}

        all_masks = []
        all_boxes = []
        all_scores = []
        all_labels = []
        all_object_ids = []

        for prompt in cc.prompts:
            if "geometric_prompt" in cc.state:
                del cc.state["geometric_prompt"]

            cc.state = cc.processor.set_text_prompt(prompt.strip(), cc.state)

            masks = cc.state.get("masks")
            boxes = cc.state.get("boxes")
            scores = cc.state.get("scores")

            if masks is not None and masks.numel() > 0:
                for i in range(len(masks)):
                    mask_np = masks[i].squeeze().cpu().numpy()
                    box = boxes[i].cpu().numpy().tolist() if boxes is not None and i < len(boxes) else None
                    score = float(scores[i].cpu()) if scores is not None and i < len(scores) else 0.0

                    # Boundary suppression
                    if cc.enable_boundary_suppression and box:
                        if is_near_boundary(box, frame.shape, cc.boundary_margin):
                            continue

                    # Hotstart
                    if cc.enable_hotstart:
                        det_hash = f"{prompt}_{int(box[0]) if box else 0}_{int(box[1]) if box else 0}"
                        if det_hash not in cc.pending_detections:
                            cc.pending_detections[det_hash] = {"frames": 1, "data": None}
                            continue
                        else:
                            cc.pending_detections[det_hash]["frames"] += 1
                            if cc.pending_detections[det_hash]["frames"] < cc.hotstart_frames:
                                continue
                            del cc.pending_detections[det_hash]

                    # Fill holes
                    if cc.enable_fill_holes:
                        mask_np = fill_holes_in_mask(mask_np, cc.fill_hole_area)

                    # Smooth edges
                    if cc.enable_smooth_edges:
                        mask_np = smooth_mask_edges(mask_np, cc.smooth_kernel_size)

                    # Persistent object IDs
                    object_id = len(all_masks)
                    if cc.enable_persistent_ids:
                        if cc.tracked_objects:
                            match_id = match_detection_to_object(
                                mask_np,
                                {oid: obj["last_mask"] for oid, obj in cc.tracked_objects.items()
                                 if "last_mask" in obj},
                                cc.iou_threshold
                            )
                            if match_id is not None:
                                object_id = match_id
                            else:
                                object_id = cc.next_object_id
                                cc.next_object_id += 1

                        if object_id not in cc.tracked_objects:
                            cc.tracked_objects[object_id] = {
                                "label": prompt.strip(),
                                "first_seen": cc.frame_count,
                                "color": COLORS[object_id % len(COLORS)],
                            }
                            cc.object_colors[object_id] = COLORS[object_id % len(COLORS)]

                        cc.tracked_objects[object_id]["last_seen"] = cc.frame_count
                        cc.tracked_objects[object_id]["last_mask"] = mask_np
                        cc.tracked_objects[object_id]["confidence"] = score

                    # Memory tracking
                    if cc.enable_memory_tracking:
                        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
                        update_memory_bank(object_id, mask_tensor)

                    # ===== YOLO INTEGRATION =====
                    yolo_info = {}

                    # YOLO Classification
                    if cc.enable_yolo_classify and box and cc.yolo_classify_model is not None:
                        if cc.frame_count % cc.yolo_classify_every_n == 0:
                            classify_result = classify_region(frame, box, prompt.strip())
                            if classify_result:
                                yolo_info["classify"] = classify_result
                                if classify_result["yolo_confidence"] >= cc.yolo_classify_threshold:
                                    cc.log(f"YOLO: {classify_result['yolo_class']} ({classify_result['yolo_confidence']:.0%})")

                    # YOLO Pose Estimation (only for person-like labels)
                    if cc.enable_yolo_pose and box and cc.yolo_pose_model is not None:
                        if is_person_label(prompt.strip()):
                            pose_result = estimate_pose(frame, box)
                            if pose_result and pose_result["confidence"] >= cc.yolo_pose_threshold:
                                yolo_info["pose"] = pose_result
                                cc.last_poses[object_id] = pose_result
                                cc.log(f"Pose detected for {prompt} (conf: {pose_result['confidence']:.0%})")

                    detection = {
                        "id": object_id,
                        "label": prompt.strip(),
                        "confidence": score,
                        "box": box,
                        "persistent_id": object_id if cc.enable_persistent_ids else None,
                        "yolo": yolo_info if yolo_info else None,
                        "tracked": False,  # Fresh detection from SAM3
                    }
                    new_detections.append(detection)

                    all_masks.append(mask_np)
                    all_object_ids.append(object_id)
                    if box:
                        all_boxes.append(box)
                    all_scores.append(score)
                    all_labels.append(prompt.strip())

        # Remove overlapping masks
        if cc.enable_non_overlap and len(all_masks) > 1:
            all_masks = remove_mask_overlaps(all_masks, all_scores)

        # Occlusion suppression
        if cc.enable_occlusion_suppression and len(all_masks) > 1:
            keep_indices = []
            for i, mask_i in enumerate(all_masks):
                is_occluded = False
                for j, mask_j in enumerate(all_masks):
                    if i != j and all_scores[j] > all_scores[i]:
                        overlap = np.logical_and(mask_i, mask_j).sum() / (mask_i.sum() + 1e-6)
                        if overlap > cc.occlusion_threshold:
                            is_occluded = True
                            break
                if not is_occluded:
                    keep_indices.append(i)

            all_masks = [all_masks[i] for i in keep_indices]
            all_boxes = [all_boxes[i] for i in keep_indices if i < len(all_boxes)]
            all_scores = [all_scores[i] for i in keep_indices]
            all_labels = [all_labels[i] for i in keep_indices]
            all_object_ids = [all_object_ids[i] for i in keep_indices]

        # Store for tracking
        if all_masks:
            cc.last_masks = torch.stack([torch.from_numpy(m).unsqueeze(0) for m in all_masks])
            cc.last_boxes = torch.tensor(all_boxes) if all_boxes else None
            cc.last_scores = torch.tensor(all_scores) if all_scores else None
            cc.last_labels = all_labels
            cc.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            cc.last_masks = None
            cc.last_boxes = None
            cc.last_scores = None
            cc.last_labels = None

        # CLIP-based visual matching filter
        if cc.visual_match_enabled and cc.reference_embedding is not None and new_detections:
            matched_detections = []
            matched_indices = []

            for i, det in enumerate(new_detections):
                box = det.get("box")
                if box:
                    # Crop the detected region
                    x1, y1, x2, y2 = [int(v) for v in box]
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                        # Get CLIP embedding
                        crop_embedding = get_clip_embedding(crop_pil)
                        if crop_embedding is not None:
                            similarity = compute_clip_similarity(cc.reference_embedding, crop_embedding)
                            det["clip_similarity"] = similarity

                            if similarity >= cc.visual_match_threshold:
                                matched_detections.append(det)
                                matched_indices.append(i)
                                cc.log(f"Visual match: {det['label']} (sim: {similarity:.2f})")

            if matched_detections:
                new_detections = matched_detections
                # Also filter masks
                if all_masks and matched_indices:
                    all_masks = [all_masks[i] for i in matched_indices if i < len(all_masks)]
                    cc.last_masks = torch.stack([torch.from_numpy(m).unsqueeze(0) for m in all_masks]) if all_masks else None
            else:
                cc.log("No visual matches found", "WARN")
                new_detections = []

        # Atomically update detections (only update if we have new detections,
        # otherwise keep the existing tracked detections)
        if new_detections:
            with cc.lock:
                cc.current_detections = new_detections
        # Note: If SAM3 found nothing but we have tracked objects, keep them
        # They will be removed by tracking when they actually leave the frame

        if all_labels:
            cc.log(f"Detected: {', '.join(all_labels)}")

    elif cc.enable_tracking and cc.last_masks is not None and not cc.paused:
        # Track with optical flow
        tracked = track_frame(frame)
        if tracked is not None:
            cc.last_masks = tracked

            # Update detections based on tracked masks and remove objects that left frame
            valid_indices = update_detections_from_tracked_masks(tracked, frame.shape)

            # If some masks were invalidated, update the mask list too
            if valid_indices is not None and len(valid_indices) < len(tracked):
                # Keep only valid masks
                valid_masks = [tracked[i] for i in valid_indices]
                if valid_masks:
                    cc.last_masks = torch.stack(valid_masks)
                else:
                    cc.last_masks = None

                # Also update labels, scores, boxes to stay in sync
                if cc.last_labels:
                    cc.last_labels = [cc.last_labels[i] for i in valid_indices if i < len(cc.last_labels)]
                if cc.last_scores is not None and len(cc.last_scores) > 0:
                    try:
                        idx_tensor = torch.tensor(valid_indices, dtype=torch.long)
                        cc.last_scores = cc.last_scores[idx_tensor] if len(valid_indices) > 0 else None
                    except Exception:
                        cc.last_scores = None
                if cc.last_boxes is not None and len(cc.last_boxes) > 0:
                    try:
                        idx_tensor = torch.tensor(valid_indices, dtype=torch.long)
                        cc.last_boxes = cc.last_boxes[idx_tensor] if len(valid_indices) > 0 else None
                    except Exception:
                        cc.last_boxes = None

    # Overlay masks on frame
    display = frame.copy()
    if cc.last_masks is not None:
        display = overlay_masks(display, cc.last_masks, cc.last_boxes, cc.last_scores, cc.last_labels)

    # Draw pose overlays
    if cc.enable_yolo_pose and cc.last_poses:
        for obj_id, pose_data in cc.last_poses.items():
            display = draw_pose_overlay(display, pose_data, obj_id)

    # Obstacle detection during navigation (run on keyframes)
    if cc.obstacle_detection_active and is_keyframe and not cc.paused:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            obstacles = detect_obstacles(frame, pil_image)

            if obstacles:
                cc.current_obstacles = obstacles
                display = overlay_obstacles(display, obstacles)

                # Log high-severity obstacles that should alert
                for obs in obstacles:
                    if obs.get("should_alert") and obs.get("type") in ["high", "medium"]:
                        cc.log(f"OBSTACLE: {obs['label']} ({obs['distance']})", "WARN")
        except Exception as e:
            cc.log(f"Obstacle overlay error: {e}", "ERROR")

    return display


def track_frame(frame: np.ndarray) -> Optional[torch.Tensor]:
    """Track masks using optical flow."""
    if cc.last_masks is None or cc.prev_gray is None:
        return None

    try:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            cc.prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        h, w = curr_gray.shape
        flow_map_x = np.arange(w).reshape(1, -1).repeat(h, axis=0).astype(np.float32)
        flow_map_y = np.arange(h).reshape(-1, 1).repeat(w, axis=1).astype(np.float32)
        flow_map_x += flow[:, :, 0]
        flow_map_y += flow[:, :, 1]

        tracked_masks = []
        for mask in cc.last_masks:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy().squeeze()
            else:
                mask_np = mask.squeeze()

            if mask_np.shape != (h, w):
                mask_np = cv2.resize(mask_np.astype(np.float32), (w, h))

            warped = cv2.remap(
                mask_np.astype(np.float32),
                flow_map_x, flow_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            warped = (warped > 0.5).astype(np.float32)

            if cc.enable_fill_holes:
                warped = fill_holes_in_mask(warped, cc.fill_hole_area)
            if cc.enable_smooth_edges:
                warped = smooth_mask_edges(warped, cc.smooth_kernel_size)

            tracked_masks.append(torch.from_numpy(warped).unsqueeze(0).to(cc.device_str))

        cc.prev_gray = curr_gray

        if tracked_masks:
            return torch.stack(tracked_masks)

    except Exception as e:
        cc.log(f"Tracking error: {e}", "ERROR")

    return None


def overlay_masks(frame: np.ndarray, masks: torch.Tensor, boxes=None, scores=None, labels=None, alpha=0.5) -> np.ndarray:
    """Overlay masks on frame."""
    if masks is None or masks.numel() == 0:
        return frame

    overlay = frame.copy()
    h, w = frame.shape[:2]
    masks_np = masks.squeeze(1).cpu().numpy()

    scores_np = scores.cpu().numpy() if scores is not None and isinstance(scores, torch.Tensor) else scores

    for i, mask in enumerate(masks_np):
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5

        # Use persistent color if available
        if cc.enable_persistent_ids and i < len(cc.current_detections):
            det = cc.current_detections[i]
            obj_id = det.get("persistent_id")
            color = cc.object_colors.get(obj_id, COLORS[i % len(COLORS)])
        else:
            color = COLORS[i % len(COLORS)]

        mask_region = mask.astype(bool)
        overlay[mask_region] = (
            overlay[mask_region] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)

        # Draw contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        # Draw label
        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)

            label = labels[i] if labels and i < len(labels) else "object"
            conf = scores_np[i] if scores_np is not None and i < len(scores_np) else 0.0

            # Add persistent ID and YOLO info to label
            text_parts = []
            if cc.enable_persistent_ids and i < len(cc.current_detections):
                obj_id = cc.current_detections[i].get("persistent_id")
                text_parts.append(f"#{obj_id}")

            text_parts.append(f"{label} {conf:.0%}")

            # Add YOLO classification if available
            if i < len(cc.current_detections):
                det = cc.current_detections[i]
                yolo_info = det.get("yolo")
                if yolo_info and "classify" in yolo_info:
                    yolo_class = yolo_info["classify"]["yolo_class"]
                    yolo_conf = yolo_info["classify"]["yolo_confidence"]
                    text_parts.append(f"[{yolo_class} {yolo_conf:.0%}]")

            text = " ".join(text_parts)

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)

            cv2.rectangle(overlay, (x, y - th - 4), (x + tw + 4, y), color, -1)
            cv2.putText(overlay, text, (x + 2, y - 2), font, 0.5, (255, 255, 255), 1)

    return overlay


def generate_frames():
    """Generator for video streaming."""
    global cc

    while cc.running:
        if cc.camera is None or not cc.camera.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = cc.camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Apply flip transformations
        if cc.flip_horizontal and cc.flip_vertical:
            frame = cv2.flip(frame, -1)  # Flip both
        elif cc.flip_horizontal:
            frame = cv2.flip(frame, 1)  # Flip horizontally (mirror)
        elif cc.flip_vertical:
            frame = cv2.flip(frame, 0)  # Flip vertically

        start = time.time()

        # Store raw frame (without overlays) for Claude analysis
        cc.current_raw_frame = frame.copy()

        # Process frame (adds overlays)
        display = process_frame(frame)

        # Calculate FPS
        elapsed = time.time() - start
        cc.fps = 1.0 / elapsed if elapsed > 0 else 0

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cc.current_frame = display
        cc.current_frame_jpeg = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + cc.current_frame_jpeg + b'\r\n')


def analyze_with_claude(image_data: str, label: str) -> str:
    """Send image to Claude for analysis."""
    global ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        return "Error: ANTHROPIC_API_KEY not set. Set it via environment variable or --api-key argument."

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"This is a cropped image of a detected '{label}'. Please provide a brief, detailed description of what you see. Focus on: appearance, distinctive features, actions/pose, and any notable details. Keep it concise (2-3 sentences)."
                        }
                    ],
                }
            ],
        )

        return message.content[0].text

    except Exception as e:
        return f"Analysis error: {str(e)}"


def analysis_worker():
    """Background worker for Claude analysis."""
    global cc

    while cc.running:
        if cc.analysis_queue:
            with cc.lock:
                if cc.analysis_queue:
                    item = cc.analysis_queue.pop(0)
                    cc.analyzing = True
                else:
                    item = None

            if item:
                cc.log(f"Analyzing object #{item['id']}...", "INFO")

                detections = cc.current_detections
                label = "object"
                for det in detections:
                    if det.get("id") == item["id"]:
                        label = det.get("label", "object")
                        break

                result = analyze_with_claude(item["image_data"], label)
                cc.add_analysis_result(item["id"], result)
                cc.log(f"Analysis complete for #{item['id']}", "SUCCESS")
                cc.analyzing = False
        else:
            time.sleep(0.5)


# ===== FLASK ROUTES =====

@app.route('/')
def index():
    """Main command center page."""
    return render_template('index.html',
                          prompts=cc.prompts,
                          threshold=cc.confidence_threshold,
                          skip_frames=cc.skip_frames,
                          tracking=cc.enable_tracking,
                          features=cc.get_feature_status(),
                          yolo_available=cc.yolo_available)


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    """Get current status."""
    filtered, hidden = cc.get_filtered_detections()
    return jsonify({
        "running": cc.running,
        "paused": cc.paused,
        "fps": round(cc.fps, 1),
        "frame_count": cc.frame_count,
        "device": cc.device_str,
        "detections": filtered,
        "hidden_counts": hidden,
        "prompts": cc.prompts,
        "max_objects": cc.max_objects_per_prompt,
        "show_all": cc.show_all_matches,
        "analyzing": cc.analyzing,
        "analysis_queue_size": len(cc.analysis_queue),
        "features": cc.get_feature_status(),
        "tracked_objects_count": len(cc.tracked_objects),
        "memory_bank_size": len(cc.memory_bank),
        "yolo_available": cc.yolo_available,
        "poses_count": len(cc.last_poses),
    })


@app.route('/api/logs')
def api_logs():
    """Get recent logs."""
    return jsonify({"logs": cc.get_logs()})


@app.route('/api/analysis_results')
def api_analysis_results():
    """Get analysis results."""
    with cc.lock:
        results = list(cc.analysis_results)
    return jsonify({"results": results})


@app.route('/api/set_prompts', methods=['POST'])
def api_set_prompts():
    """Set detection prompts."""
    data = request.json
    prompts_str = data.get("prompts", "object")
    cc.prompts = [p.strip() for p in prompts_str.split(",") if p.strip()]
    cc.state = None
    cc.last_masks = None
    cc.last_boxes = None
    cc.last_scores = None
    cc.last_labels = None
    cc.tracked_objects = {}
    cc.memory_bank = {}
    cc.last_poses = {}
    cc.log(f"Prompts updated: {', '.join(cc.prompts)}")
    return jsonify({"success": True, "prompts": cc.prompts})


@app.route('/api/set_limit', methods=['POST'])
def api_set_limit():
    """Set max objects limit for a prompt."""
    data = request.json
    prompt = data.get("prompt")
    limit = data.get("limit")

    if limit is not None:
        cc.max_objects_per_prompt[prompt] = int(limit)
    elif prompt in cc.max_objects_per_prompt:
        del cc.max_objects_per_prompt[prompt]

    cc.log(f"Limit for '{prompt}': {limit if limit else 'unlimited'}")
    return jsonify({"success": True})


@app.route('/api/toggle_show_all', methods=['POST'])
def api_toggle_show_all():
    """Toggle show all matches for a prompt."""
    data = request.json
    prompt = data.get("prompt")
    cc.show_all_matches[prompt] = not cc.show_all_matches.get(prompt, False)
    cc.log(f"Show all for '{prompt}': {cc.show_all_matches[prompt]}")
    return jsonify({"success": True, "show_all": cc.show_all_matches[prompt]})


@app.route('/api/toggle_pause', methods=['POST'])
def api_toggle_pause():
    """Toggle pause state."""
    cc.paused = not cc.paused
    cc.log("Paused" if cc.paused else "Resumed")
    return jsonify({"success": True, "paused": cc.paused})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset detection state."""
    cc.state = None
    cc.last_masks = None
    cc.last_boxes = None
    cc.last_scores = None
    cc.last_labels = None
    cc.tracked_objects = {}
    cc.memory_bank = {}
    cc.object_colors = {}
    cc.next_object_id = 1
    cc.pending_detections = {}
    cc.last_poses = {}
    cc.clear_detections()
    cc.log("Detection state reset")
    return jsonify({"success": True})


@app.route('/api/set_threshold', methods=['POST'])
def api_set_threshold():
    """Set confidence threshold."""
    data = request.json
    cc.confidence_threshold = float(data.get("threshold", 0.3))
    if cc.processor:
        cc.processor.confidence_threshold = cc.confidence_threshold
    cc.log(f"Threshold set to {cc.confidence_threshold:.2f}")
    return jsonify({"success": True})


@app.route('/api/set_skip_frames', methods=['POST'])
def api_set_skip_frames():
    """Set skip frames value."""
    data = request.json
    cc.skip_frames = max(1, int(data.get("skip_frames", 3)))
    cc.log(f"Skip frames set to {cc.skip_frames}")
    return jsonify({"success": True})


# ===== FEATURE TOGGLE ROUTES =====

@app.route('/api/toggle_feature', methods=['POST'])
def api_toggle_feature():
    """Toggle a feature on/off."""
    data = request.json
    feature = data.get("feature")

    feature_map = {
        "tracking": "enable_tracking",
        "memory_tracking": "enable_memory_tracking",
        "persistent_ids": "enable_persistent_ids",
        "fill_holes": "enable_fill_holes",
        "non_overlap": "enable_non_overlap",
        "smooth_edges": "enable_smooth_edges",
        "boundary_suppression": "enable_boundary_suppression",
        "occlusion_suppression": "enable_occlusion_suppression",
        "hotstart": "enable_hotstart",
        "yolo_classify": "enable_yolo_classify",
        "yolo_pose": "enable_yolo_pose",
        "show_keypoint_labels": "show_keypoint_labels",
        "show_skeleton": "show_skeleton",
        "label_spoofing": "enable_label_spoofing",
    }

    if feature in feature_map:
        attr = feature_map[feature]
        current = getattr(cc, attr)
        setattr(cc, attr, not current)
        new_val = getattr(cc, attr)
        cc.log(f"{feature}: {'ON' if new_val else 'OFF'}")
        return jsonify({"success": True, "feature": feature, "enabled": new_val})

    return jsonify({"success": False, "error": "Unknown feature"})


@app.route('/api/set_feature_param', methods=['POST'])
def api_set_feature_param():
    """Set a feature parameter value."""
    data = request.json
    param = data.get("param")
    value = data.get("value")

    param_map = {
        "fill_hole_area": ("fill_hole_area", int),
        "smooth_kernel_size": ("smooth_kernel_size", int),
        "boundary_margin": ("boundary_margin", int),
        "occlusion_threshold": ("occlusion_threshold", float),
        "hotstart_frames": ("hotstart_frames", int),
        "iou_threshold": ("iou_threshold", float),
        "memory_max_frames": ("memory_max_frames", int),
        "yolo_classify_threshold": ("yolo_classify_threshold", float),
        "yolo_pose_threshold": ("yolo_pose_threshold", float),
        "yolo_classify_every_n": ("yolo_classify_every_n", int),
        "keypoint_radius": ("keypoint_radius", int),
        "skeleton_thickness": ("skeleton_thickness", int),
    }

    if param in param_map:
        attr, type_fn = param_map[param]
        setattr(cc, attr, type_fn(value))
        cc.log(f"{param} set to {value}")
        return jsonify({"success": True})

    return jsonify({"success": False, "error": "Unknown parameter"})


@app.route('/api/analyze_object', methods=['POST'])
def api_analyze_object():
    """Queue an object for Claude analysis with mask-based cropping."""
    data = request.json
    detection_id = data.get("detection_id")
    box = data.get("box")
    mask_index = data.get("mask_index")  # Index into cc.last_masks

    # Use raw frame (without overlays) for analysis
    if cc.current_raw_frame is None:
        return jsonify({"success": False, "error": "No frame available"})

    try:
        frame = cc.current_raw_frame.copy()
        h, w = frame.shape[:2]

        # Try to use mask for better cropping
        mask = None
        if mask_index is not None and cc.last_masks is not None:
            try:
                if mask_index < len(cc.last_masks):
                    mask = cc.last_masks[mask_index].squeeze().cpu().numpy()
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
            except Exception as e:
                cc.log(f"Could not get mask: {e}", "WARN")
                mask = None

        if mask is not None and mask.sum() > 0:
            # Use mask to create a clean crop with transparent/white background
            # Get bounding box from mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Add padding
            pad = 15
            x1 = max(0, x_min - pad)
            y1 = max(0, y_min - pad)
            x2 = min(w, x_max + pad)
            y2 = min(h, y_max + pad)

            # Crop the region
            crop = frame[y1:y2, x1:x2].copy()
            mask_crop = mask[y1:y2, x1:x2]

            # Apply mask - set background to white for cleaner analysis
            mask_3ch = np.stack([mask_crop] * 3, axis=-1)
            crop = np.where(mask_3ch, crop, 255).astype(np.uint8)

        elif box:
            # Fallback to box-based cropping
            x1, y1, x2, y2 = [int(v) for v in box]
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame

        _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_data = base64.b64encode(buffer).decode('utf-8')

        cc.queue_analysis(detection_id, image_data)
        cc.log(f"Queued object #{detection_id} for analysis (mask-cropped: {mask is not None})")

        return jsonify({"success": True})

    except Exception as e:
        cc.log(f"Failed to queue analysis: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/describe_scene', methods=['POST'])
def api_describe_scene():
    """Send full scene to Claude for description."""
    global ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        return jsonify({"success": False, "error": "ANTHROPIC_API_KEY not set"})

    if cc.current_raw_frame is None:
        return jsonify({"success": False, "error": "No frame available"})

    try:
        import anthropic

        frame = cc.current_raw_frame.copy()
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_data = base64.b64encode(buffer).decode('utf-8')

        cc.log("Analyzing full scene with Claude...")

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Please describe this scene in detail. Include: the setting/environment, all visible objects and people, their positions and relationships, any activities or actions taking place, lighting conditions, and any notable details. Be comprehensive but concise (3-5 sentences)."
                        }
                    ],
                }
            ],
        )

        result = message.content[0].text
        cc.log("Scene analysis complete", "SUCCESS")

        # Add to analysis results
        cc.add_analysis_result(-1, f"[SCENE] {result}")

        return jsonify({
            "success": True,
            "description": result
        })

    except Exception as e:
        cc.log(f"Scene analysis failed: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/tracked_objects')
def api_tracked_objects():
    """Get list of tracked objects with persistent IDs."""
    objects = []
    for obj_id, data in cc.tracked_objects.items():
        objects.append({
            "id": obj_id,
            "label": data.get("label"),
            "first_seen": data.get("first_seen"),
            "last_seen": data.get("last_seen"),
            "confidence": data.get("confidence", 0),
            "frames_tracked": data.get("last_seen", 0) - data.get("first_seen", 0),
        })
    return jsonify({"objects": objects})


@app.route('/api/poses')
def api_poses():
    """Get current pose data for all detected persons."""
    poses = []
    for obj_id, pose_data in cc.last_poses.items():
        poses.append({
            "object_id": obj_id,
            "confidence": pose_data.get("confidence", 0),
            "keypoints": [
                {"name": name, "x": kp[0], "y": kp[1], "confidence": kp[2]}
                for name, kp in zip(POSE_KEYPOINTS, pose_data.get("keypoints", []))
            ]
        })
    return jsonify({"poses": poses})


@app.route('/api/coco_mapping')
def api_coco_mapping():
    """Get SAM3 to COCO label mapping."""
    return jsonify({
        "mapping": SAM3_TO_COCO,
        "coco_classes": COCO_CLASSES
    })


# ===== VOICE SEARCH ROUTES =====

def parse_voice_query_with_claude(voice_text: str) -> Dict:
    """
    Use Claude to parse a voice query into search prompts.

    Handles queries like:
    - "help me find a red car"
    - "can you search for a person and a dog"
    - "look for my phone, keys, and wallet"
    - "find the blue cup on the table"

    Returns dict with:
        - prompts: List of parsed object prompts (comma-separated format)
        - is_multi: Whether multiple objects were requested
        - feedback: Human-readable feedback message
    """
    global ANTHROPIC_API_KEY

    if not ANTHROPIC_API_KEY:
        return {
            "success": False,
            "error": "ANTHROPIC_API_KEY not set",
            "prompts": [voice_text],  # Fallback: use raw text
            "feedback": f"API key not set. Searching for: {voice_text}"
        }

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": f"""Parse this voice command for an object detection system. Extract the objects the user wants to find.

Voice command: "{voice_text}"

Rules:
1. Extract object names/descriptions that can be detected visually
2. If multiple objects are mentioned, list them all
3. Include color/size descriptors if mentioned (e.g., "red car", "large dog")
4. Ignore filler words like "help me find", "can you search for", "look for"
5. Return ONLY a JSON object, no other text

Return JSON format:
{{"prompts": ["object1", "object2"], "feedback": "Searching for object1 and object2"}}

Examples:
- "help me find a red car" -> {{"prompts": ["red car"], "feedback": "Searching for red car"}}
- "search for people and dogs" -> {{"prompts": ["person", "dog"], "feedback": "Searching for person and dog"}}
- "find my phone and keys" -> {{"prompts": ["phone", "keys"], "feedback": "Searching for phone and keys"}}
- "look for a blue cup" -> {{"prompts": ["blue cup"], "feedback": "Searching for blue cup"}}"""
                }
            ],
        )

        response_text = message.content[0].text.strip()

        # Parse JSON from response
        # Handle potential markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        prompts = result.get("prompts", [])
        feedback = result.get("feedback", f"Searching for {', '.join(prompts)}")

        return {
            "success": True,
            "prompts": prompts,
            "is_multi": len(prompts) > 1,
            "feedback": feedback,
            "raw_query": voice_text
        }

    except json.JSONDecodeError as e:
        cc.log(f"Failed to parse Claude response as JSON: {e}", "ERROR")
        # Fallback: just use the voice text directly
        return {
            "success": True,
            "prompts": [voice_text],
            "is_multi": False,
            "feedback": f"Searching for {voice_text}",
            "raw_query": voice_text
        }
    except Exception as e:
        cc.log(f"Voice query parsing error: {e}", "ERROR")
        return {
            "success": False,
            "error": str(e),
            "prompts": [],
            "feedback": "Failed to parse voice command"
        }


def check_describe_command(voice_text: str) -> Optional[Dict]:
    """
    Check if voice command is a describe command.
    Returns dict with action info if it's a describe command, None otherwise.

    Handles:
    - "describe scene" / "describe the scene" / "what do you see"
    - "describe object 1" / "describe the first object" / "tell me about object 2"
    - "analyze object 3" / "what is object 1"
    """
    text_lower = voice_text.lower().strip()

    # Scene describe patterns
    scene_patterns = [
        "describe scene", "describe the scene", "describe this scene",
        "what do you see", "what's in the scene", "describe everything",
        "describe the view", "describe what you see", "analyze scene",
        "tell me about the scene", "what's happening"
    ]

    for pattern in scene_patterns:
        if pattern in text_lower:
            return {
                "action": "describe_scene",
                "feedback": "Describing the scene..."
            }

    # Object describe patterns - extract object number
    import re

    # Patterns like "describe object 1", "analyze object 2", "tell me about object 3"
    object_patterns = [
        r"describe (?:the )?(?:object|item|thing) (\d+)",
        r"analyze (?:the )?(?:object|item|thing) (\d+)",
        r"what is (?:object|item|thing) (\d+)",
        r"tell me about (?:object|item|thing) (\d+)",
        r"describe (?:the )?(\d+)(?:st|nd|rd|th)? (?:object|item|thing)",
        r"describe number (\d+)",
        r"object (\d+) describe",
    ]

    for pattern in object_patterns:
        match = re.search(pattern, text_lower)
        if match:
            obj_num = int(match.group(1))
            return {
                "action": "describe_object",
                "object_id": obj_num,
                "feedback": f"Describing object {obj_num}..."
            }

    # Ordinal patterns like "describe the first object", "analyze the second item"
    ordinals = {
        "first": 0, "1st": 0,
        "second": 1, "2nd": 1,
        "third": 2, "3rd": 2,
        "fourth": 3, "4th": 3,
        "fifth": 4, "5th": 4,
    }

    for ordinal, idx in ordinals.items():
        if ordinal in text_lower and ("object" in text_lower or "item" in text_lower or "thing" in text_lower):
            if "describe" in text_lower or "analyze" in text_lower or "tell me" in text_lower or "what is" in text_lower:
                return {
                    "action": "describe_object",
                    "object_index": idx,
                    "feedback": f"Describing the {ordinal} object..."
                }

    return None


@app.route('/api/voice_search', methods=['POST'])
def api_voice_search():
    """Process a voice search query through Claude and set prompts."""
    data = request.json
    voice_text = data.get("text", "").strip()

    if not voice_text:
        return jsonify({"success": False, "error": "No voice text provided"})

    cc.log(f"Voice query received: '{voice_text}'", "INFO")
    cc.last_voice_query = voice_text

    # First check for describe commands
    describe_cmd = check_describe_command(voice_text)
    if describe_cmd:
        cc.add_voice_feedback(describe_cmd["feedback"], "info")

        if describe_cmd["action"] == "describe_scene":
            return jsonify({
                "success": True,
                "action": "describe_scene",
                "feedback": describe_cmd["feedback"],
                "tts_message": describe_cmd["feedback"]
            })

        elif describe_cmd["action"] == "describe_object":
            # Find the object to describe
            obj_id = describe_cmd.get("object_id")
            obj_index = describe_cmd.get("object_index")

            detections = cc.current_detections

            if not detections:
                return jsonify({
                    "success": False,
                    "error": "No objects detected",
                    "feedback": "No objects are currently detected"
                })

            # Find the detection
            target_det = None
            target_index = None

            if obj_id is not None:
                # Look for object with this ID
                for i, det in enumerate(detections):
                    if det.get("id") == obj_id:
                        target_det = det
                        target_index = i
                        break
            elif obj_index is not None:
                # Use index directly
                if obj_index < len(detections):
                    target_det = detections[obj_index]
                    target_index = obj_index

            if target_det is None:
                return jsonify({
                    "success": False,
                    "error": f"Object not found",
                    "feedback": f"Could not find the specified object"
                })

            return jsonify({
                "success": True,
                "action": "describe_object",
                "detection": target_det,
                "mask_index": target_index,
                "feedback": describe_cmd["feedback"],
                "tts_message": describe_cmd["feedback"]
            })

    # Parse the voice query with Claude for search
    result = parse_voice_query_with_claude(voice_text)

    if result["success"] and result["prompts"]:
        # Update prompts
        cc.prompts = result["prompts"]
        cc.last_parsed_prompts = result["prompts"]

        # Reset detection state for new search
        cc.state = None
        cc.last_masks = None
        cc.last_boxes = None
        cc.last_scores = None
        cc.last_labels = None
        cc.tracked_objects = {}
        cc.memory_bank = {}
        cc.last_poses = {}

        prompt_str = ", ".join(result["prompts"])
        cc.log(f"Voice search: {prompt_str}", "SUCCESS")
        cc.add_voice_feedback(result["feedback"], "success")

        return jsonify({
            "success": True,
            "action": "search",
            "prompts": result["prompts"],
            "prompt_string": prompt_str,
            "is_multi": result["is_multi"],
            "feedback": result["feedback"],
            "tts_message": result["feedback"]
        })
    else:
        error_msg = result.get("error", "Could not understand the voice command")
        cc.add_voice_feedback(f"Error: {error_msg}", "error")
        return jsonify({
            "success": False,
            "error": error_msg,
            "feedback": result.get("feedback", "Failed to process voice command")
        })


@app.route('/api/voice_feedback')
def api_voice_feedback():
    """Get recent voice feedback messages."""
    with cc.lock:
        messages = list(cc.voice_feedback_messages)
    return jsonify({
        "messages": messages,
        "last_query": cc.last_voice_query,
        "last_prompts": cc.last_parsed_prompts
    })


@app.route('/api/toggle_voice', methods=['POST'])
def api_toggle_voice():
    """Toggle voice features."""
    data = request.json
    feature = data.get("feature", "voice")

    if feature == "voice":
        cc.voice_enabled = not cc.voice_enabled
        cc.log(f"Voice input: {'ON' if cc.voice_enabled else 'OFF'}")
        return jsonify({"success": True, "enabled": cc.voice_enabled})
    elif feature == "tts":
        cc.tts_enabled = not cc.tts_enabled
        cc.log(f"TTS output: {'ON' if cc.tts_enabled else 'OFF'}")
        return jsonify({"success": True, "enabled": cc.tts_enabled})

    return jsonify({"success": False, "error": "Unknown feature"})


# ===== CAMERA ROUTES =====

@app.route('/api/cameras')
def api_cameras():
    """Get list of available cameras."""
    cameras = detect_available_cameras()
    cc.available_cameras = cameras
    return jsonify({
        "cameras": cameras,
        "current_camera": cc.current_camera_id,
        "flip_horizontal": cc.flip_horizontal,
        "flip_vertical": cc.flip_vertical
    })


@app.route('/api/switch_camera', methods=['POST'])
def api_switch_camera():
    """Switch to a different camera."""
    data = request.json
    camera_id = data.get("camera_id")

    if camera_id is None:
        return jsonify({"success": False, "error": "No camera_id provided"})

    camera_id = int(camera_id)

    success = switch_camera(camera_id)

    return jsonify({
        "success": success,
        "current_camera": cc.current_camera_id,
        "message": f"Switched to camera {camera_id}" if success else f"Failed to switch to camera {camera_id}"
    })


@app.route('/api/flip_camera', methods=['POST'])
def api_flip_camera():
    """Toggle camera flip (horizontal/vertical)."""
    data = request.json
    direction = data.get("direction", "horizontal")

    if direction == "horizontal":
        cc.flip_horizontal = not cc.flip_horizontal
        cc.log(f"Horizontal flip: {'ON' if cc.flip_horizontal else 'OFF'}")
        # Reset detection state when flip changes
        reset_detection_state()
        return jsonify({
            "success": True,
            "flip_horizontal": cc.flip_horizontal,
            "flip_vertical": cc.flip_vertical
        })
    elif direction == "vertical":
        cc.flip_vertical = not cc.flip_vertical
        cc.log(f"Vertical flip: {'ON' if cc.flip_vertical else 'OFF'}")
        # Reset detection state when flip changes
        reset_detection_state()
        return jsonify({
            "success": True,
            "flip_horizontal": cc.flip_horizontal,
            "flip_vertical": cc.flip_vertical
        })
    elif direction == "both":
        cc.flip_horizontal = not cc.flip_horizontal
        cc.flip_vertical = not cc.flip_vertical
        cc.log(f"Flip both: H={'ON' if cc.flip_horizontal else 'OFF'}, V={'ON' if cc.flip_vertical else 'OFF'}")
        reset_detection_state()
        return jsonify({
            "success": True,
            "flip_horizontal": cc.flip_horizontal,
            "flip_vertical": cc.flip_vertical
        })

    return jsonify({"success": False, "error": "Invalid direction"})


@app.route('/api/set_flip', methods=['POST'])
def api_set_flip():
    """Set flip state explicitly."""
    data = request.json
    flip_h = data.get("flip_horizontal")
    flip_v = data.get("flip_vertical")

    changed = False

    if flip_h is not None and flip_h != cc.flip_horizontal:
        cc.flip_horizontal = bool(flip_h)
        changed = True

    if flip_v is not None and flip_v != cc.flip_vertical:
        cc.flip_vertical = bool(flip_v)
        changed = True

    if changed:
        cc.log(f"Flip set: H={'ON' if cc.flip_horizontal else 'OFF'}, V={'ON' if cc.flip_vertical else 'OFF'}")
        reset_detection_state()

    return jsonify({
        "success": True,
        "flip_horizontal": cc.flip_horizontal,
        "flip_vertical": cc.flip_vertical
    })


# ===== REFERENCE IMAGE SEARCH API =====

@app.route('/api/upload_reference', methods=['POST'])
def api_upload_reference():
    """
    Upload a reference image for search.
    Modes:
    - 'description': Use Claude to describe, then search by text
    - 'visual': Use CLIP for visual similarity matching
    """
    global cc

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"})

    mode = request.form.get('mode', 'description')  # 'description' or 'visual'

    try:
        file = request.files['image']
        image_data = file.read()

        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        cc.reference_image = pil_image

        # Get base64 for Claude
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=90)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        if mode == 'description':
            # Use Claude to describe the image
            cc.log("Analyzing reference image with Claude...")
            description = describe_image_with_claude(base64_image)

            if description:
                cc.reference_description = description
                cc.visual_match_enabled = False

                # Set as prompt
                cc.prompts = [description]
                cc.state = None
                cc.last_masks = None
                reset_detection_state()

                cc.log(f"Reference search: '{description}'", "SUCCESS")

                return jsonify({
                    "success": True,
                    "mode": "description",
                    "description": description,
                    "prompt": description
                })
            else:
                return jsonify({"success": False, "error": "Failed to describe image"})

        elif mode == 'visual':
            # Use CLIP for visual matching
            if not cc.clip_available:
                return jsonify({
                    "success": False,
                    "error": "CLIP not available. Install with: pip install transformers"
                })

            cc.log("Computing CLIP embedding for reference image...")
            embedding = get_clip_embedding(pil_image)

            if embedding is not None:
                cc.reference_embedding = embedding
                cc.visual_match_enabled = True

                # Also get a description for display
                description = describe_image_with_claude(base64_image)
                cc.reference_description = description or "Visual reference"

                # Set a generic prompt to detect objects
                cc.prompts = ["object"]
                cc.state = None
                cc.last_masks = None
                reset_detection_state()

                cc.log(f"Visual matching enabled for: {cc.reference_description}", "SUCCESS")

                return jsonify({
                    "success": True,
                    "mode": "visual",
                    "description": cc.reference_description,
                    "message": "Visual matching enabled"
                })
            else:
                return jsonify({"success": False, "error": "Failed to compute CLIP embedding"})

        else:
            return jsonify({"success": False, "error": f"Unknown mode: {mode}"})

    except Exception as e:
        cc.log(f"Reference upload failed: {e}", "ERROR")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/clear_reference', methods=['POST'])
def api_clear_reference():
    """Clear the reference image."""
    global cc

    cc.reference_image = None
    cc.reference_embedding = None
    cc.reference_description = None
    cc.visual_match_enabled = False

    cc.log("Reference image cleared")

    return jsonify({"success": True})


@app.route('/api/reference_status')
def api_reference_status():
    """Get reference image status."""
    return jsonify({
        "has_reference": cc.reference_image is not None,
        "description": cc.reference_description,
        "visual_match_enabled": cc.visual_match_enabled,
        "clip_available": cc.clip_available,
        "threshold": cc.visual_match_threshold
    })


# ===== GEOMETRIC PROMPTS (DRAW TO SEARCH) API =====

@app.route('/api/draw_prompt', methods=['POST'])
def api_draw_prompt():
    """
    Set a geometric prompt (box or point) from user drawing.
    This will be processed on the next frame.
    """
    global cc

    data = request.json
    prompt_type = data.get('type', 'box')  # 'box' or 'point'

    if prompt_type == 'box':
        x1 = data.get('x1')
        y1 = data.get('y1')
        x2 = data.get('x2')
        y2 = data.get('y2')

        if all(v is not None for v in [x1, y1, x2, y2]):
            cc.pending_box_prompt = (float(x1), float(y1), float(x2), float(y2))
            cc.pending_point_prompt = None
            cc.draw_mode = 'box'
            cc.log(f"Box prompt set: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})")

            return jsonify({
                "success": True,
                "type": "box",
                "box": [x1, y1, x2, y2]
            })
        else:
            return jsonify({"success": False, "error": "Invalid box coordinates"})

    elif prompt_type == 'point':
        x = data.get('x')
        y = data.get('y')

        if x is not None and y is not None:
            cc.pending_point_prompt = (float(x), float(y))
            cc.pending_box_prompt = None
            cc.draw_mode = 'point'
            cc.log(f"Point prompt set: ({x:.0f}, {y:.0f})")

            return jsonify({
                "success": True,
                "type": "point",
                "point": [x, y]
            })
        else:
            return jsonify({"success": False, "error": "Invalid point coordinates"})

    else:
        return jsonify({"success": False, "error": f"Unknown prompt type: {prompt_type}"})


@app.route('/api/clear_draw_prompt', methods=['POST'])
def api_clear_draw_prompt():
    """Clear any pending geometric prompts."""
    global cc

    cc.pending_box_prompt = None
    cc.pending_point_prompt = None
    cc.draw_mode = None

    cc.log("Draw prompt cleared")

    return jsonify({"success": True})


# ===== NAVIGATION SYSTEM API =====

@app.route('/api/navigation/start', methods=['POST'])
def api_navigation_start():
    """Start navigation to a detected object."""
    global cc

    data = request.json
    target_label = data.get("target_label") or data.get("label")
    target_id = data.get("target_id") or data.get("detection_id")

    if not target_label and target_id is None:
        return jsonify({"success": False, "error": "No target specified"})

    # Check for location memory first (from SQLite)
    memory = cc.recall_location(target_label) if target_label else None
    memory_hint = None
    if memory:
        memory_hint = f"I remember finding {target_label} in the {memory.get('context', 'unknown location')} before."

    cc.navigation_active = True
    cc.navigation_target = target_label
    cc.navigation_target_id = target_id
    cc.navigation_start_time = time.time()
    cc.navigation_last_seen = None
    cc.navigation_reached = False
    cc.navigation_target_history = []

    # Start obstacle detection
    cc.obstacle_detection_active = True
    cc.current_obstacles = []
    cc.obstacle_masks = None
    cc.obstacle_boxes = None

    # Create navigation session in database
    if cc.session_id:
        cc.navigation_db_id = db.start_navigation_session(cc.session_id, target_label, target_id)
        db.log_event(cc.session_id, "navigation_start", f"Started navigation to {target_label}",
                     data={"target_label": target_label, "target_id": target_id})

    # Analyze scene context
    if cc.current_raw_frame is not None:
        try:
            _, buffer = cv2.imencode('.jpg', cc.current_raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            image_data = base64.b64encode(buffer).decode('utf-8')
            cc.navigation_context = analyze_scene_context(image_data)
        except Exception as e:
            cc.log(f"Scene context analysis failed: {e}", "WARN")
            cc.navigation_context = None

    cc.log(f"Navigation started: looking for '{target_label}'", "SUCCESS")

    # Initial message
    location = cc.navigation_context.get("location", "this area") if cc.navigation_context else "this area"
    initial_message = f"Starting navigation to find {target_label}. You appear to be in {location}."
    if memory_hint:
        initial_message += f" {memory_hint}"

    return jsonify({
        "success": True,
        "target": target_label,
        "initial_message": initial_message,
        "memory_hint": memory_hint,
        "context": cc.navigation_context
    })


@app.route('/api/navigation/stop', methods=['POST'])
def api_navigation_stop():
    """Stop navigation."""
    global cc

    was_active = cc.navigation_active
    target = cc.navigation_target
    reached = cc.navigation_reached

    # If we reached the target, remember its location (in SQLite)
    if reached and cc.navigation_context and target:
        location = cc.navigation_context.get("location", "unknown location")
        cc.remember_location(target, location)

    # End navigation session in database
    if cc.navigation_db_id:
        db.end_navigation_session(
            cc.navigation_db_id,
            reached=reached,
            path_history=cc.navigation_target_history,
            scene_context=cc.navigation_context
        )
        if cc.session_id:
            db.log_event(cc.session_id, "navigation_stop",
                        f"Navigation to {target} {'reached' if reached else 'cancelled'}",
                        data={"target": target, "reached": reached})

    # Stop obstacle detection
    cc.obstacle_detection_active = False
    cc.current_obstacles = []
    cc.obstacle_masks = None
    cc.obstacle_boxes = None

    cc.navigation_active = False
    cc.navigation_target = None
    cc.navigation_target_id = None
    cc.navigation_db_id = None
    cc.navigation_start_time = None
    cc.navigation_last_seen = None
    cc.navigation_reached = False
    cc.navigation_context = None
    cc.navigation_target_history = []

    if was_active:
        cc.log(f"Navigation ended for '{target}'")

    return jsonify({
        "success": True,
        "reached": reached,
        "show_post_nav_dialog": was_active  # Tell UI to show continue/pause dialog
    })


@app.route('/api/navigation/status')
def api_navigation_status():
    """Get current navigation status and guidance."""
    status = get_navigation_status()

    # Add TTS guidance if needed
    if status.get("active") and status.get("guidance"):
        current_time = time.time()
        guidance_text = status["guidance"].get("guidance_text", "")

        # Only speak if enough time has passed and guidance changed
        if (current_time - cc.navigation_last_guidance_time > cc.navigation_guidance_interval and
            guidance_text != cc.navigation_last_guidance):
            status["speak_guidance"] = True
            cc.navigation_last_guidance = guidance_text
            cc.navigation_last_guidance_time = current_time
        else:
            status["speak_guidance"] = False

    # Add obstacle alerts
    if cc.current_obstacles:
        obstacles_for_alert = []
        for obs in cc.current_obstacles:
            if obs.get("should_alert"):
                obstacles_for_alert.append({
                    "label": obs["label"],
                    "type": obs["type"],
                    "distance": obs["distance"],
                    "alert_text": f"Watch out! {obs['label']} {obs['distance'].replace('_', ' ')}"
                })
        status["obstacles"] = obstacles_for_alert

    return jsonify(status)


@app.route('/api/navigation/analyze_scene', methods=['POST'])
def api_navigation_analyze_scene():
    """Analyze current scene for navigation context."""
    global cc

    if cc.current_raw_frame is None:
        return jsonify({"success": False, "error": "No frame available"})

    try:
        _, buffer = cv2.imencode('.jpg', cc.current_raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        image_data = base64.b64encode(buffer).decode('utf-8')
        context = analyze_scene_context(image_data)

        if context:
            cc.navigation_context = context
            return jsonify({"success": True, "context": context})
        else:
            return jsonify({"success": False, "error": "Analysis failed"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/location_memory')
def api_location_memory():
    """Get stored location memory (from SQLite)."""
    memories = cc.get_all_location_memories()
    return jsonify({
        "success": True,
        "memory": memories
    })


@app.route('/api/location_memory/recall', methods=['POST'])
def api_recall_location():
    """Recall where an object was last found (from SQLite)."""
    data = request.json
    label = data.get("label", "")

    memory = cc.recall_location(label)

    if memory:
        return jsonify({
            "success": True,
            "found": True,
            "label": label,
            "location": memory.get("context"),
            "frequency": memory.get("frequency", 1),
            "last_seen": memory.get("last_seen")
        })
    else:
        return jsonify({
            "success": True,
            "found": False,
            "label": label,
            "message": f"No memory of where {label} was found"
        })


@app.route('/api/location_memory/clear', methods=['POST'])
def api_clear_location_memory():
    """Clear location memory."""
    data = request.json or {}
    label = data.get("label")

    cc.clear_location_memory(label)

    return jsonify({
        "success": True,
        "message": f"Cleared location memory" + (f" for {label}" if label else "")
    })


# ===== OBSTACLE DETECTION API =====

@app.route('/api/navigation/obstacles')
def api_navigation_obstacles():
    """Get current obstacles detected during navigation."""
    return jsonify({
        "success": True,
        "obstacles": cc.current_obstacles,
        "active": cc.obstacle_detection_active
    })


# ===== DATABASE HISTORY API =====

@app.route('/api/history/detections')
def api_history_detections():
    """Get detection history from database."""
    label = request.args.get('label')
    limit = int(request.args.get('limit', 100))

    history = db.get_detection_history(session_id=cc.session_id, label=label, limit=limit)

    return jsonify({
        "success": True,
        "detections": history,
        "count": len(history)
    })


@app.route('/api/history/analysis')
def api_history_analysis():
    """Get analysis history from database."""
    limit = int(request.args.get('limit', 50))

    history = db.get_analysis_history(session_id=cc.session_id, limit=limit)

    return jsonify({
        "success": True,
        "analyses": history,
        "count": len(history)
    })


@app.route('/api/history/navigation')
def api_history_navigation():
    """Get navigation history from database."""
    limit = int(request.args.get('limit', 20))

    history = db.get_navigation_history(session_id=cc.session_id, limit=limit)

    return jsonify({
        "success": True,
        "navigations": history,
        "count": len(history)
    })


@app.route('/api/session/stats')
def api_session_stats():
    """Get statistics for the current session."""
    if not cc.session_id:
        return jsonify({"success": False, "error": "No active session"})

    stats = db.get_session_stats(cc.session_id)

    return jsonify({
        "success": True,
        "session_id": cc.session_id,
        "stats": stats
    })


def generate_self_signed_cert(cert_dir: str = None) -> Tuple[str, str]:
    """Generate a self-signed SSL certificate for HTTPS."""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime

        if cert_dir is None:
            cert_dir = os.path.join(os.path.dirname(__file__), '.ssl')

        os.makedirs(cert_dir, exist_ok=True)

        key_path = os.path.join(cert_dir, 'key.pem')
        cert_path = os.path.join(cert_dir, 'cert.pem')

        # Check if certs already exist
        if os.path.exists(key_path) and os.path.exists(cert_path):
            print(f"Using existing SSL certificates from {cert_dir}")
            return cert_path, key_path

        print("Generating self-signed SSL certificate...")

        # Generate private key
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SAM3 Command Center"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("127.0.0.1"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(key, hashes.SHA256(), default_backend())

        # Write key
        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # Write certificate
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        print(f"SSL certificate generated: {cert_path}")
        return cert_path, key_path

    except ImportError:
        print("WARNING: cryptography package not installed. Cannot generate SSL certificate.")
        print("  Install with: pip install cryptography")
        print("  Or provide --ssl-cert and --ssl-key arguments")
        return None, None


def main():
    global cc

    parser = argparse.ArgumentParser(description="SAM3 Web Command Center")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera device ID")
    parser.add_argument("--device", "-d", type=str, default=None, help="Device (cuda, mps, cpu)")
    parser.add_argument("--prompt", type=str, default="object", help="Initial prompts (comma-separated)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--skip-frames", type=int, default=3, help="Process every N frames")
    parser.add_argument("--no-tracking", action="store_true", help="Disable optical flow tracking")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO models")
    parser.add_argument("--api-key", type=str, default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--no-https", action="store_true", help="Disable HTTPS (not recommended - microphone won't work)")
    parser.add_argument("--ssl-cert", type=str, default=None, help="Path to SSL certificate file")
    parser.add_argument("--ssl-key", type=str, default=None, help="Path to SSL private key file")

    args = parser.parse_args()

    # Set API key from argument if provided
    global ANTHROPIC_API_KEY
    if args.api_key:
        ANTHROPIC_API_KEY = args.api_key
        print("Using API key from command line argument")
    elif ANTHROPIC_API_KEY:
        print("Using API key from environment variable")
    else:
        print("WARNING: No Anthropic API key set. Claude features (analysis, voice search) will not work.")
        print("  Set via: --api-key YOUR_KEY or ANTHROPIC_API_KEY=YOUR_KEY")

    # Configure command center
    cc.prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]
    cc.confidence_threshold = args.threshold
    cc.skip_frames = args.skip_frames
    cc.enable_tracking = not args.no_tracking

    if args.device:
        cc.device_str = args.device

    # Create database session
    cc.session_id = db.create_session(
        device=args.device or "auto",
        prompts=cc.prompts,
        settings={
            "threshold": args.threshold,
            "skip_frames": args.skip_frames,
            "tracking": not args.no_tracking,
            "yolo": not args.no_yolo
        }
    )
    cc.log(f"Database session started: {cc.session_id[:8]}...")

    # Load model
    load_model(args.checkpoint)

    # Skip YOLO if requested
    if args.no_yolo:
        cc.yolo_available = False
        cc.log("YOLO disabled via command line")

    # Detect available cameras
    cc.log("Detecting available cameras...")
    cc.available_cameras = detect_available_cameras()
    cc.log(f"Found {len(cc.available_cameras)} camera(s)", "SUCCESS")
    for cam in cc.available_cameras:
        cc.log(f"  Camera {cam['id']}: {cam['description']}")

    # Open camera
    cc.log(f"Opening camera {args.camera}...")
    cc.camera = cv2.VideoCapture(args.camera)
    cc.current_camera_id = args.camera

    if not cc.camera.isOpened():
        cc.log(f"Failed to open camera {args.camera}", "ERROR")
        return

    width = int(cc.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cc.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cc.log(f"Camera opened: {width}x{height}", "SUCCESS")

    cc.running = True

    # Start analysis worker
    analysis_thread = threading.Thread(target=analysis_worker, daemon=True)
    analysis_thread.start()

    print(f"\n{'='*50}")
    print(f"SAM3 Web Command Center")
    print(f"{'='*50}")

    # Setup SSL (HTTPS is default, use --no-https to disable)
    ssl_context = None
    protocol = "http"

    if not args.no_https:
        if args.ssl_cert and args.ssl_key:
            # Use provided certificates
            if os.path.exists(args.ssl_cert) and os.path.exists(args.ssl_key):
                ssl_context = (args.ssl_cert, args.ssl_key)
                protocol = "https"
                print(f"Using provided SSL certificates")
            else:
                print(f"ERROR: SSL certificate files not found")
                print(f"  Cert: {args.ssl_cert}")
                print(f"  Key: {args.ssl_key}")
                return
        else:
            # Generate self-signed certificate
            cert_path, key_path = generate_self_signed_cert()
            if cert_path and key_path:
                ssl_context = (cert_path, key_path)
                protocol = "https"
                print(f"Using auto-generated self-signed certificate")
                print(f"  NOTE: You may need to accept the security warning in your browser")
            else:
                print("WARNING: Could not setup HTTPS. Falling back to HTTP.")
                print("  Microphone and navigation features may not work without HTTPS!")
    else:
        print("WARNING: HTTPS disabled. Microphone and navigation features may not work!")

    print(f"Open {protocol}://localhost:{args.port} in your browser")
    print(f"YOLO: {'Available' if cc.yolo_available else 'Not available'}")
    print(f"CLIP: {'Available' if cc.clip_available else 'Not available'}")
    if protocol == "https":
        print(f"HTTPS: Enabled (microphone and navigation available)")
    else:
        print(f"HTTPS: Disabled (use default or remove --no-https for full features)")
    print(f"{'='*50}\n")

    try:
        if ssl_context:
            app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False, ssl_context=ssl_context)
        else:
            app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)
    finally:
        cc.running = False
        if cc.camera:
            cc.camera.release()


if __name__ == "__main__":
    main()
