"""
Enhanced Video Processing with Multiple Detection Methods
Includes:
1. Multi-scale face detection
2. Face quality assessment
3. Temporal consistency checking
4. Artifact detection
5. Frame quality filtering
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import os

# Initialize multiple face detectors for robustness
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

def assess_frame_quality(frame: np.ndarray) -> float:
    """
    Assess frame quality using blur detection
    
    Args:
        frame: Input frame
    
    Returns:
        Quality score (higher is better)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Laplacian variance (measures blur)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Brightness
    brightness = np.mean(gray)
    
    # Contrast
    contrast = gray.std()
    
    # Combined quality score
    quality = (laplacian_var / 100) * (contrast / 50) * (1 - abs(brightness - 128) / 128)
    
    return quality

def detect_faces_multi_scale(
    frame: np.ndarray,
    min_face_size: int = 50
) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces using multiple scales and methods
    
    Args:
        frame: Input frame (RGB)
        min_face_size: Minimum face size
    
    Returns:
        List of face bounding boxes (x, y, w, h)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    all_faces = []
    
    # Try multiple scale factors
    for scale_factor in [1.05, 1.1, 1.2]:
        for min_neighbors in [3, 4, 5]:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(min_face_size, min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                all_faces.extend(faces)
    
    # Try profile detection if no frontal faces found
    if len(all_faces) == 0:
        profiles = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size)
        )
        all_faces.extend(profiles)
    
    # Remove duplicate detections using NMS
    if len(all_faces) > 0:
        all_faces = non_max_suppression(np.array(all_faces), 0.3)
    
    return all_faces

def non_max_suppression(boxes: np.ndarray, overlap_thresh: float = 0.3) -> List:
    """
    Apply non-maximum suppression to remove overlapping boxes
    
    Args:
        boxes: Array of boxes (x, y, w, h)
        overlap_thresh: Overlap threshold
    
    Returns:
        Filtered list of boxes
    """
    if len(boxes) == 0:
        return []
    
    # Convert to (x1, y1, x2, y2)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = boxes[:, 2] * boxes[:, 3]
    indices = np.argsort(areas)[::-1]
    
    keep = []
    
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
        
        # Compute IoU
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / areas[indices[1:]]
        
        indices = indices[1:][overlap <= overlap_thresh]
    
    return boxes[keep].tolist()

def verify_face_with_eyes(face_region: np.ndarray) -> bool:
    """
    Verify if detected region contains a face by checking for eyes
    
    Args:
        face_region: Cropped face region (RGB)
    
    Returns:
        True if eyes are detected
    """
    gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20)
    )
    
    return len(eyes) >= 2

def extract_frames_smart(
    video_path: str,
    num_frames: int = 30,
    quality_threshold: float = 10.0
) -> Tuple[List[np.ndarray], Dict]:
    """
    Extract high-quality frames from video
    
    Args:
        video_path: Path to video
        num_frames: Target number of frames
        quality_threshold: Minimum quality score
    
    Returns:
        List of frames and metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        raise ValueError("Video has no frames")
    
    # Sample more frames than needed
    sample_size = min(total_frames, num_frames * 3)
    frame_indices = np.linspace(0, total_frames - 1, sample_size, dtype=int)
    
    frames_with_quality = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            quality = assess_frame_quality(frame_rgb)
            
            if quality >= quality_threshold:
                frames_with_quality.append((frame_rgb, quality, idx))
    
    cap.release()
    
    # Sort by quality and take top frames
    frames_with_quality.sort(key=lambda x: x[1], reverse=True)
    selected_frames = [f[0] for f in frames_with_quality[:num_frames]]
    
    metadata = {
        'total_frames': total_frames,
        'fps': fps,
        'selected_frames': len(selected_frames),
        'avg_quality': np.mean([f[1] for f in frames_with_quality[:num_frames]]) if frames_with_quality else 0
    }
    
    print(f"✓ Extracted {len(selected_frames)} high-quality frames (avg quality: {metadata['avg_quality']:.2f})")
    
    return selected_frames, metadata

def detect_and_crop_faces(
    frames: List[np.ndarray],
    target_size: int = 224,
    verify_with_eyes: bool = True
) -> Tuple[List[np.ndarray], Dict]:
    """
    Detect and crop faces with quality verification
    
    Args:
        frames: List of frames
        target_size: Target face size
        verify_with_eyes: Whether to verify faces by detecting eyes
    
    Returns:
        List of face crops and detection statistics
    """
    face_crops = []
    stats = {
        'frames_processed': len(frames),
        'faces_detected': 0,
        'faces_verified': 0,
        'detection_confidence': []
    }
    
    for i, frame in enumerate(frames):
        # Detect faces
        faces = detect_faces_multi_scale(frame)
        
        if len(faces) == 0:
            print(f"  No face in frame {i}")
            continue
        
        stats['faces_detected'] += len(faces)
        
        # Get largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Add padding
        padding = int(max(w, h) * 0.3)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Crop face
        face_crop = frame[y:y+h, x:x+w]
        
        # Verify face quality
        if verify_with_eyes:
            if verify_face_with_eyes(face_crop):
                stats['faces_verified'] += 1
            else:
                print(f"  Face in frame {i} failed eye verification")
                # Still use it but note the issue
        
        # Resize to target size
        face_crop = cv2.resize(face_crop, (target_size, target_size))
        face_crops.append(face_crop)
        
        # Calculate confidence based on face size
        face_area = w * h
        frame_area = frame.shape[0] * frame.shape[1]
        confidence = min(1.0, face_area / (frame_area * 0.1))
        stats['detection_confidence'].append(confidence)
    
    # Fallback: use center crops if no faces detected
    if len(face_crops) == 0:
        print("⚠ No faces detected, using center crops as fallback")
        for frame in frames[:20]:
            center_crop = crop_center(frame, target_size)
            face_crops.append(center_crop)
        stats['fallback_used'] = True
    else:
        stats['fallback_used'] = False
    
    stats['avg_confidence'] = np.mean(stats['detection_confidence']) if stats['detection_confidence'] else 0
    
    print(f"✓ Detected {len(face_crops)} faces (verified: {stats['faces_verified']}, avg confidence: {stats['avg_confidence']:.2f})")
    
    return face_crops, stats

def crop_center(image: np.ndarray, size: int) -> np.ndarray:
    """Center crop and resize image"""
    h, w = image.shape[:2]
    crop_size = min(h, w)
    
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    
    cropped = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
    cropped = cv2.resize(cropped, (size, size))
    
    return cropped

def analyze_temporal_consistency(face_crops: List[np.ndarray]) -> Dict:
    """
    Analyze temporal consistency across frames
    Deepfakes often have inconsistencies between frames
    
    Args:
        face_crops: List of face images
    
    Returns:
        Dictionary with consistency metrics
    """
    if len(face_crops) < 2:
        return {'consistency_score': 1.0, 'warning': 'Not enough frames'}
    
    # Calculate frame-to-frame differences
    differences = []
    
    for i in range(len(face_crops) - 1):
        img1 = face_crops[i].astype(float)
        img2 = face_crops[i + 1].astype(float)
        
        # Mean squared difference
        mse = np.mean((img1 - img2) ** 2)
        differences.append(mse)
    
    # Calculate statistics
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    # High variance in differences might indicate manipulation
    consistency_score = 1.0 / (1.0 + std_diff / (mean_diff + 1e-6))
    
    return {
        'consistency_score': consistency_score,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'suspicious': std_diff > mean_diff * 2
    }

def detect_compression_artifacts(image: np.ndarray) -> Dict:
    """
    Detect compression artifacts that might indicate manipulation
    
    Args:
        image: Input image
    
    Returns:
        Dictionary with artifact metrics
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Detect blocking artifacts (8x8 DCT blocks)
    h, w = gray.shape
    block_differences = []
    
    for i in range(8, h - 8, 8):
        for j in range(8, w - 8, 8):
            # Difference across block boundaries
            vert_diff = abs(int(gray[i, j]) - int(gray[i-1, j]))
            horiz_diff = abs(int(gray[i, j]) - int(gray[i, j-1]))
            block_differences.append(vert_diff + horiz_diff)
    
    avg_block_diff = np.mean(block_differences) if block_differences else 0
    
    return {
        'edge_density': edge_density,
        'block_artifacts': avg_block_diff,
        'suspicious': avg_block_diff > 20
    }
