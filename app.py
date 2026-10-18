from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import re
from datetime import datetime
import sqlite3
import json
from collections import defaultdict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize models
reader = easyocr.Reader(['en'], gpu=False)
model = YOLO('yolov8n.pt')

# Vehicle class names
VEHICLE_CLASSES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Alert list (suspicious/VIP plates)
ALERT_PLATES = ['KA01AB1234', 'MH02CD5678']  # Add plates to monitor

# Initialize database
def init_db():
    conn = sqlite3.connect('traffic.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  plate TEXT,
                  vehicle_type TEXT,
                  confidence REAL,
                  timestamp TEXT,
                  source TEXT,
                  alert INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

init_db()

def save_detection(plate, vehicle_type, confidence, source='video'):
    conn = sqlite3.connect('traffic.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    alert = 1 if plate in ALERT_PLATES else 0
    c.execute('INSERT INTO detections (plate, vehicle_type, confidence, timestamp, source, alert) VALUES (?, ?, ?, ?, ?, ?)',
              (plate, vehicle_type, confidence, timestamp, source, alert))
    conn.commit()
    conn.close()

def enhance_plate_image(image):
    """Apply fast preprocessing to enhance blurred plates"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Quick 3-step enhancement
    enhanced_images = []
    
    # 1. CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_images.append(clahe.apply(gray))
    
    # 2. Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    enhanced_images.append(sharpened)
    
    # 3. Upscale if small
    if gray.shape[0] < 80:
        upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        enhanced_images.append(upscaled)
    
    return enhanced_images
    text = text.upper().replace(' ', '').replace('-', '').replace('.', '')
    corrections = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'Q': '0', 'D': '0', '|': '1', 'L': '1'}
    
    result = ""
    for i, char in enumerate(text):
        if i < 2:
            result += 'O' if char == '0' else ('I' if char == '1' else char)
        elif i < 4:
            result += corrections.get(char, char) if char.isalpha() else char
        elif i < len(text) - 4:
            result += char
        else:
            result += corrections.get(char, char) if char.isalpha() else char
    return result

def is_valid_indian_plate(text):
    text = text.upper().replace(' ', '')
    if len(text) < 8 or len(text) > 12:
        return False
    has_letters = sum(c.isalpha() for c in text) >= 2
    has_numbers = sum(c.isdigit() for c in text) >= 4
    return has_letters and has_numbers

def process_frame(frame):
    """Process single frame and return detections - AGGRESSIVE PLATE DETECTION"""
    detections = []
    detected_plates_in_frame = set()
    
    # FIRST: Try direct OCR on entire frame (catches plates without vehicle detection)
    try:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_full = enhance_plate_image(frame)
        
        for enhanced_img in enhanced_full:
            ocr_results = reader.readtext(enhanced_img, detail=1, paragraph=False)
            
            for detection in ocr_results:
                bbox, text, confidence = detection
                text_clean = ''.join(c for c in text if c.isalnum()).upper()
                text_corrected = clean_plate_text(text_clean)
                
                if confidence > 0.2 and len(text_corrected) >= 7 and is_valid_indian_plate(text_corrected):
                    if text_corrected not in detected_plates_in_frame:
                        detected_plates_in_frame.add(text_corrected)
                        
                        # Draw detection on frame
                        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
                        is_alert = text_corrected in ALERT_PLATES
                        color = (0, 0, 255) if is_alert else (0, 255, 0)
                        
                        cv2.polylines(frame, [pts], True, color, 3)
                        x, y = int(bbox[0][0]), int(bbox[0][1])
                        
                        label = f"{text_corrected}"
                        if is_alert:
                            label += " [ALERT!]"
                        
                        # Draw text with background
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                        cv2.rectangle(frame, (x, y-45), (x + text_size[0] + 10, y-5), color, -1)
                        cv2.putText(frame, label, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                        
                        detections.append({
                            'plate': text_corrected,
                            'vehicle_type': 'Unknown',
                            'confidence': round(confidence * 100, 2),
                            'alert': is_alert
                        })
                        break
            if len(detected_plates_in_frame) > 0:
                break
    except Exception as e:
        pass
    
    # SECOND: Detect vehicles and look for plates in them
    results = model(frame, classes=[2, 3, 5, 7], conf=0.2, verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            vehicle_type = VEHICLE_CLASSES.get(cls, 'Unknown')
            
            # Draw vehicle box
            color = (255, 0, 0) if vehicle_type == 'Motorcycle' else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, vehicle_type, (x1, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            h = y2 - y1
            
            # Try multiple regions
            regions = [
                (max(0, y1 + int(h*0.6)), y2, x1, x2),  # Bottom 40%
                (max(0, y1 + int(h*0.5)), y2, x1, x2),  # Bottom 50%
                (y1, min(frame.shape[0], y1 + int(h*0.3)), x1, x2),  # Top 30%
            ]
            
            best_plate = None
            best_confidence = 0
            
            for ry1, ry2, rx1, rx2 in regions:
                plate_region = frame[ry1:ry2, rx1:rx2]
                
                if plate_region.size < 100:
                    continue
                
                enhanced_images = enhance_plate_image(plate_region)
                
                for enhanced_img in enhanced_images:
                    try:
                        ocr_results = reader.readtext(enhanced_img, detail=1, paragraph=False)
                        
                        for detection in ocr_results:
                            _, text, confidence = detection
                            text_clean = ''.join(c for c in text if c.isalnum()).upper()
                            text_corrected = clean_plate_text(text_clean)
                            
                            if confidence > 0.2 and len(text_corrected) >= 7 and is_valid_indian_plate(text_corrected):
                                if confidence > best_confidence and text_corrected not in detected_plates_in_frame:
                                    best_confidence = confidence
                                    best_plate = {
                                        'plate': text_corrected,
                                        'vehicle_type': vehicle_type,
                                        'confidence': round(confidence * 100, 2),
                                    }
                                    break
                    except:
                        continue
                    
                    if best_plate:
                        break
                
                if best_plate:
                    break
            
            if best_plate:
                detected_plates_in_frame.add(best_plate['plate'])
                is_alert = best_plate['plate'] in ALERT_PLATES
                box_color = (0, 0, 255) if is_alert else (0, 255, 0)
                
                label = f"{best_plate['plate']}"
                if is_alert:
                    label += " [ALERT!]"
                
                # Thick box for plate detection
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                cv2.rectangle(frame, (x1, y1-45), (x1 + text_size[0] + 10, y1-5), box_color, -1)
                cv2.putText(frame, label, (x1+5, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                best_plate['alert'] = is_alert
                detections.append(best_plate)
    
    # Add detection count overlay
    if len(detections) > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, frame.shape[0]-80), (300, frame.shape[0]-10), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, f"DETECTED: {len(detections)} PLATES", (20, frame.shape[0]-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame, detections

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_detections = []
    vehicle_counts = defaultdict(int)
    frame_count = 0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process only 5 seconds max for speed
    max_frames = min(total_frames, fps * 5 if fps > 0 else 150)
    
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {max_frames} frames (5 seconds)...")
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 15th frame for speed
        if frame_count % 15 == 0:
            processed_frame, detections = process_frame(frame)
            
            for det in detections:
                vehicle_counts[det['vehicle_type']] += 1
                if det['plate'] not in [d['plate'] for d in all_detections]:
                    all_detections.append(det)
                    save_detection(det['plate'], det['vehicle_type'], det['confidence'], 'video')
            
            frame = processed_frame
            print(f"✓ Frame {frame_count}/{max_frames} - Found {len(all_detections)} plates")
        
        # Add statistics overlay
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {frame_count}/{max_frames}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Plates: {len(all_detections)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Cars: {vehicle_counts['Car']}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Bikes: {vehicle_counts['Motorcycle']}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"✅ Complete! Found {len(all_detections)} plates")
    return all_detections, 'output_video.mp4', dict(vehicle_counts)

# Webcam streaming
camera = None

def generate_frames():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        processed_frame, detections = process_frame(frame.copy())
        
        for det in detections:
            save_detection(det['plate'], det['vehicle_type'], det['confidence'], 'webcam')
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/webcam')
def webcam_page():
    return render_template('webcam.html')

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process', methods=['POST'])
def process():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        plates, output_video, vehicle_counts = process_video(filepath)
        return jsonify({
            'success': True,
            'plates': plates,
            'output_video': output_video,
            'vehicle_counts': vehicle_counts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    conn = sqlite3.connect('traffic.db')
    c = conn.cursor()
    
    # Total detections
    c.execute('SELECT COUNT(*) FROM detections')
    total = c.fetchone()[0]
    
    # Alerts
    c.execute('SELECT COUNT(*) FROM detections WHERE alert=1')
    alerts = c.fetchone()[0]
    
    # Vehicle types
    c.execute('SELECT vehicle_type, COUNT(*) FROM detections GROUP BY vehicle_type')
    vehicle_data = dict(c.fetchall())
    
    # Recent detections
    c.execute('SELECT plate, vehicle_type, confidence, timestamp, alert FROM detections ORDER BY id DESC LIMIT 20')
    recent = [{'plate': r[0], 'vehicle_type': r[1], 'confidence': r[2], 'timestamp': r[3], 'alert': r[4]} for r in c.fetchall()]
    
    # Hourly distribution
    c.execute('SELECT strftime("%H", timestamp) as hour, COUNT(*) FROM detections GROUP BY hour')
    hourly = dict(c.fetchall())
    
    conn.close()
    
    return jsonify({
        'total': total,
        'alerts': alerts,
        'vehicle_data': vehicle_data,
        'recent': recent,
        'hourly': hourly
    })

@app.route('/api/search')
def search_plate():
    query = request.args.get('q', '')
    conn = sqlite3.connect('traffic.db')
    c = conn.cursor()
    c.execute('SELECT plate, vehicle_type, confidence, timestamp, alert FROM detections WHERE plate LIKE ? ORDER BY id DESC',
              (f'%{query}%',))
    results = [{'plate': r[0], 'vehicle_type': r[1], 'confidence': r[2], 'timestamp': r[3], 'alert': r[4]} for r in c.fetchall()]
    conn.close()
    return jsonify(results)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)