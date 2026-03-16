# Zhenning
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import os


class ROIsExtractor:
    """
    Extract Mouth ROI and Full-Face ROI, generating 96x96 and 88x88 outputs
    """
    
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # Use the official FaceMesh lip set (set of edges, extract all endpoints)
        lip_indices = set()
        for edge in self.mp_face_mesh.FACEMESH_LIPS:
            lip_indices.add(edge[0])
            lip_indices.add(edge[1])
        self.mouth_indices = sorted(list(lip_indices))
        
        # Smoothing filter state
        self.smoothed_mouth_bbox = None
        self.smoothed_face_bbox = None
        self.alpha = 0.3  # Smoothing coefficient
        
    def _get_face_bbox(self, landmarks, img_w, img_h):
        """Compute the full-face bounding box from landmarks"""
        x_coords = []
        y_coords = []
        for landmark in landmarks.landmark:
            x_coords.append(landmark.x * img_w)
            y_coords.append(landmark.y * img_h)
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Expand bbox to a square and add margin
        w = x_max - x_min
        h = y_max - y_min
        
        # Use the larger side as the square size
        size = max(w, h)
        # Expand by 1.2x to ensure the full face is included
        size = int(size * 1.2)
        
        # Compute center point
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        # Compute square boundaries
        x_min = int(cx - size / 2)
        y_min = int(cy - size / 2)
        x_max = int(cx + size / 2)
        y_max = int(cy + size / 2)
        
        return (x_min, y_min, x_max, y_max)
    
    def _get_mouth_bbox(self, landmarks, img_w, img_h):
        """Compute the mouth bounding box from landmarks (using official lip indices)"""
        mouth_points = []
        for idx in self.mouth_indices:
            landmark = landmarks.landmark[idx]
            px, py = landmark.x * img_w, landmark.y * img_h
            mouth_points.append([px, py])
        
        mouth_points = np.array(mouth_points)
        x_min, y_min = mouth_points.min(axis=0)
        x_max, y_max = mouth_points.max(axis=0)
        
        # Compute tight bbox
        w = x_max - x_min
        h = y_max - y_min
        
        # Expand by 1.6x (common ratio)
        scale = 1.6
        w_expanded = w * scale
        h_expanded = h * scale
        
        # Or use margin-based expansion
        # margin = 0.2 * max(w, h)
        # w_expanded = w + 2 * margin
        # h_expanded = h + 2 * margin
        
        # Compute center point
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        # Compute square boundaries (use the larger side)
        size = max(w_expanded, h_expanded)
        
        x_min = int(cx - size / 2)
        y_min = int(cy - size / 2)
        x_max = int(cx + size / 2)
        y_max = int(cy + size / 2)
        
        return (x_min, y_min, x_max, y_max)
    
    def _smooth_bbox(self, bbox, bbox_type='mouth'):
        """Smooth the bounding box using exponential moving average"""
        if bbox_type == 'mouth':
            smoothed_bbox = self.smoothed_mouth_bbox
        else:
            smoothed_bbox = self.smoothed_face_bbox
        
        if smoothed_bbox is None:
            # First detection, return the original bbox directly
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            new_smoothed = (cx, cy, w, h)
        else:
            # Apply exponential moving average
            cx_old, cy_old, w_old, h_old = smoothed_bbox
            cx_new = (bbox[0] + bbox[2]) / 2
            cy_new = (bbox[1] + bbox[3]) / 2
            w_new = bbox[2] - bbox[0]
            h_new = bbox[3] - bbox[1]
            
            cx = self.alpha * cx_new + (1 - self.alpha) * cx_old
            cy = self.alpha * cy_new + (1 - self.alpha) * cy_old
            w = self.alpha * w_new + (1 - self.alpha) * w_old
            h = self.alpha * h_new + (1 - self.alpha) * h_old
            
            new_smoothed = (cx, cy, w, h)
        
        # Update state
        if bbox_type == 'mouth':
            self.smoothed_mouth_bbox = new_smoothed
        else:
            self.smoothed_face_bbox = new_smoothed
        
        # Convert back to (x_min, y_min, x_max, y_max)
        x_min = int(cx - w/2)
        y_min = int(cy - h/2)
        x_max = int(cx + w/2)
        y_max = int(cy + h/2)
        
        return (x_min, y_min, x_max, y_max)
    
    def _clip_bbox(self, bbox, img_w, img_h):
        """Clip bbox to image boundaries and ensure validity"""
        x_min, y_min, x_max, y_max = bbox
        
        # Clip to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w - 1, x_max)
        y_max = min(img_h - 1, y_max)
        
        # Ensure width and height are at least 1
        if x_max <= x_min:
            x_max = x_min + 1
        if y_max <= y_min:
            y_max = y_min + 1
        
        return (x_min, y_min, x_max, y_max)
    
    def _crop_and_resize(self, frame, bbox, target_size):
        """Crop the bbox region from frame and resize it to target_size"""
        x1, y1, x2, y2 = bbox
        
        # Crop ROI
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            # Return a black image
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Resize
        roi_resized = cv2.resize(roi, (target_size, target_size))
        
        return roi_resized
    
    def _random_crop_88(self, roi_96, offset=None):
        """
        Randomly crop from 96x96 to 88x88 (for training)
        
        Args:
            roi_96: 96x96 input image
            offset: optional fixed offset (top, left); if provided, use this offset
        
        Returns:
            88x88 cropped image
        """
        if offset is not None:
            top, left = offset
        else:
            top = np.random.randint(0, 96 - 88 + 1)
            left = np.random.randint(0, 96 - 88 + 1)
        
        roi_88 = roi_96[top:top+88, left:left+88]
        return roi_88
    
    def _center_crop_88(self, roi_96):
        """Center crop from 96x96 to 88x88 (for validation/testing)"""
        top = (96 - 88) // 2
        left = (96 - 88) // 2
        roi_88 = roi_96[top:top+88, left:left+88]
        return roi_88
    
    def process_video(self, input_video_path, output_dir, mode='train'):
        """
        Process a single video file and generate all outputs
        
        Args:
            input_video_path: input video path
            output_dir: output directory
            mode: 'train' or 'eval' (determines the 88x88 cropping strategy)
        
        Returns:
            dict: dictionary of output file paths
        """
        # Create a video output directory (use video name as subdirectory)
        video_name = Path(input_video_path).stem
        video_output_dir = Path(output_dir) / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open input video
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize output video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        outputs = {
            'mouth_96': cv2.VideoWriter(str(video_output_dir / 'mouth_96.mp4'), fourcc, fps, (96, 96)),
            'face_96': cv2.VideoWriter(str(video_output_dir / 'face_96.mp4'), fourcc, fps, (96, 96)),
            'mouth_88': cv2.VideoWriter(str(video_output_dir / 'mouth_88.mp4'), fourcc, fps, (88, 88)),
            'face_88': cv2.VideoWriter(str(video_output_dir / 'face_88.mp4'), fourcc, fps, (88, 88)),
            'overlay_debug': cv2.VideoWriter(str(video_output_dir / 'overlay_debug.mp4'), fourcc, fps, (width, height))
        }
        
        # Reset smoothing state
        self.smoothed_mouth_bbox = None
        self.smoothed_face_bbox = None
        
        # If in train mode, generate a fixed random offset for the whole video
        # This ensures all frames in the same video use the same crop position, avoiding frame-to-frame jitter
        if mode == 'train':
            fixed_crop_offset = (
                np.random.randint(0, 96 - 88 + 1),  # top
                np.random.randint(0, 96 - 88 + 1)   # left
            )
        else:
            fixed_crop_offset = None
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            mouth_bbox = None
            face_bbox = None
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get mouth bbox
                mouth_bbox = self._get_mouth_bbox(face_landmarks, width, height)
                
                # Get full-face bbox
                face_bbox = self._get_face_bbox(face_landmarks, width, height)
            
            # Smooth or fill mouth bbox
            if mouth_bbox is not None:
                mouth_bbox_smoothed = self._smooth_bbox(mouth_bbox, 'mouth')
            elif self.smoothed_mouth_bbox is not None:
                # Fill with previous frame
                cx, cy, w, h = self.smoothed_mouth_bbox
                mouth_bbox_smoothed = (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))
            else:
                # Detection failed on the first frame, use a default box (center of the image)
                cx, cy = width // 2, height // 2
                default_size = min(width, height) // 4
                mouth_bbox_smoothed = (cx - default_size//2, cy - default_size//2, 
                                       cx + default_size//2, cy + default_size//2)
            
            # Smooth or fill face bbox
            if face_bbox is not None:
                face_bbox_smoothed = self._smooth_bbox(face_bbox, 'face')
            elif self.smoothed_face_bbox is not None:
                # Fill with previous frame
                cx, cy, w, h = self.smoothed_face_bbox
                face_bbox_smoothed = (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))
            else:
                # Detection failed on the first frame, use a default box
                cx, cy = width // 2, height // 2
                default_size = int(min(width, height) * 0.8)
                face_bbox_smoothed = (cx - default_size//2, cy - default_size//2,
                                      cx + default_size//2, cy + default_size//2)
            
            # Clip to image boundaries
            mouth_bbox_clipped = self._clip_bbox(mouth_bbox_smoothed, width, height)
            face_bbox_clipped = self._clip_bbox(face_bbox_smoothed, width, height)
            
            # Generate mouth_96 and face_96
            mouth_96 = self._crop_and_resize(frame, mouth_bbox_clipped, 96)
            face_96 = self._crop_and_resize(frame, face_bbox_clipped, 96)
            
            # Generate mouth_88 and face_88
            if mode == 'train':
                # Use a fixed offset to ensure consistent crop position across all frames in the same video
                mouth_88 = self._random_crop_88(mouth_96, offset=fixed_crop_offset)
                face_88 = self._random_crop_88(face_96, offset=fixed_crop_offset)
            else:
                mouth_88 = self._center_crop_88(mouth_96)
                face_88 = self._center_crop_88(face_96)
            
            # Write output videos
            outputs['mouth_96'].write(mouth_96)
            outputs['face_96'].write(face_96)
            outputs['mouth_88'].write(mouth_88)
            outputs['face_88'].write(face_88)
            
            # Generate overlay_debug
            overlay_frame = frame.copy()
            
            # Draw mouth bbox (green)
            x1, y1, x2, y2 = mouth_bbox_clipped
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay_frame, 'Mouth', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw face bbox (blue)
            x1, y1, x2, y2 = face_bbox_clipped
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(overlay_frame, 'Face', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Add frame number and timestamp
            time_sec = frame_idx / fps if fps > 0 else 0
            text = f'Frame: {frame_idx} | Time: {time_sec:.2f}s | FPS: {fps}'
            cv2.putText(overlay_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            outputs['overlay_debug'].write(overlay_frame)
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        for writer in outputs.values():
            writer.release()
        
        print(f'Processed: {input_video_path} ({frame_idx} frames) -> {video_output_dir}')
        
        return {
            'mouth_96': str(video_output_dir / 'mouth_96.mp4'),
            'face_96': str(video_output_dir / 'face_96.mp4'),
            'mouth_88': str(video_output_dir / 'mouth_88.mp4'),
            'face_88': str(video_output_dir / 'face_88.mp4'),
            'overlay_debug': str(video_output_dir / 'overlay_debug.mp4')
        }
    
    def process_directory(self, input_dir, output_dir, mode='train', pattern='**/*.mp4'):
        """
        Batch process all videos in a directory
        
        Args:
            input_dir: input directory
            output_dir: output directory
            mode: 'train' or 'eval'
            pattern: file matching pattern (default: recursively search all mp4 files)
        
        Returns:
            list: list of all output file paths
        """
        input_path = Path(input_dir)
        video_files = list(input_path.glob(pattern))
        
        print(f'Found {len(video_files)} videos in {input_dir}')
        
        all_outputs = []
        for i, video_file in enumerate(video_files, 1):
            print(f'\n[{i}/{len(video_files)}] Processing: {video_file.name}')
            outputs = self.process_video(str(video_file), output_dir, mode)
            if outputs:
                all_outputs.append(outputs)
        
        print(f'\nCompleted! Processed {len(all_outputs)} videos.')
        return all_outputs


# Usage example
if __name__ == '__main__':
    extractor = ROIsExtractor()
    
    # Process a single file
    # outputs = extractor.process_video('path/to/video.mp4', './output', mode='train')
    
    # Batch process a directory
    # all_outputs = extractor.process_directory('path/to/dataset', './output', mode='train')
    
    print("ROI Extractor ready!")
    print("Usage:")
    print("  extractor.process_video('video.mp4', './output', mode='train')")
    print("  extractor.process_directory('./videos', './output', mode='eval')")

