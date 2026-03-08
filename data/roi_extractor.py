import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import os


class ROIsExtractor:
    """
    提取 Mouth ROI 和 Full-Face ROI，生成 96x96 和 88x88 输出
    """
    
    def __init__(self, static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
        # 使用 FaceMesh 官方嘴唇集合（边的集合，提取所有端点）
        lip_indices = set()
        for edge in self.mp_face_mesh.FACEMESH_LIPS:
            lip_indices.add(edge[0])
            lip_indices.add(edge[1])
        self.mouth_indices = sorted(list(lip_indices))
        
        # 平滑滤波器状态
        self.smoothed_mouth_bbox = None
        self.smoothed_face_bbox = None
        self.alpha = 0.3  # 平滑系数
        
    def _get_face_bbox(self, landmarks, img_w, img_h):
        """从 landmarks 计算完整人脸边界框"""
        x_coords = []
        y_coords = []
        for landmark in landmarks.landmark:
            x_coords.append(landmark.x * img_w)
            y_coords.append(landmark.y * img_h)
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 扩展 bbox 到正方形，并添加 margin
        w = x_max - x_min
        h = y_max - y_min
        
        # 使用较大的边作为正方形边长
        size = max(w, h)
        # 扩大 1.2 倍，确保包含完整人脸
        size = int(size * 1.2)
        
        # 计算中心点
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        # 计算正方形边界
        x_min = int(cx - size / 2)
        y_min = int(cy - size / 2)
        x_max = int(cx + size / 2)
        y_max = int(cy + size / 2)
        
        return (x_min, y_min, x_max, y_max)
    
    def _get_mouth_bbox(self, landmarks, img_w, img_h):
        """从 landmarks 计算嘴部边界框（使用官方嘴唇索引）"""
        mouth_points = []
        for idx in self.mouth_indices:
            landmark = landmarks.landmark[idx]
            px, py = landmark.x * img_w, landmark.y * img_h
            mouth_points.append([px, py])
        
        mouth_points = np.array(mouth_points)
        x_min, y_min = mouth_points.min(axis=0)
        x_max, y_max = mouth_points.max(axis=0)
        
        # 计算 tight bbox
        w = x_max - x_min
        h = y_max - y_min
        
        # 扩大 1.6 倍（常用比例）
        scale = 1.6
        w_expanded = w * scale
        h_expanded = h * scale
        
        # 或者使用 margin 方式
        # margin = 0.2 * max(w, h)
        # w_expanded = w + 2 * margin
        # h_expanded = h + 2 * margin
        
        # 计算中心点
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        # 计算正方形边界（取较大边）
        size = max(w_expanded, h_expanded)
        
        x_min = int(cx - size / 2)
        y_min = int(cy - size / 2)
        x_max = int(cx + size / 2)
        y_max = int(cy + size / 2)
        
        return (x_min, y_min, x_max, y_max)
    
    def _smooth_bbox(self, bbox, bbox_type='mouth'):
        """使用指数移动平均平滑边界框"""
        if bbox_type == 'mouth':
            smoothed_bbox = self.smoothed_mouth_bbox
        else:
            smoothed_bbox = self.smoothed_face_bbox
        
        if smoothed_bbox is None:
            # 首次检测，直接返回原始 bbox
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            new_smoothed = (cx, cy, w, h)
        else:
            # 应用指数移动平均
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
        
        # 更新状态
        if bbox_type == 'mouth':
            self.smoothed_mouth_bbox = new_smoothed
        else:
            self.smoothed_face_bbox = new_smoothed
        
        # 转换回 (x_min, y_min, x_max, y_max)
        x_min = int(cx - w/2)
        y_min = int(cy - h/2)
        x_max = int(cx + w/2)
        y_max = int(cy + h/2)
        
        return (x_min, y_min, x_max, y_max)
    
    def _clip_bbox(self, bbox, img_w, img_h):
        """裁剪 bbox 到图像边界内，并确保有效性"""
        x_min, y_min, x_max, y_max = bbox
        
        # 裁剪到图像边界
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w - 1, x_max)
        y_max = min(img_h - 1, y_max)
        
        # 确保宽高至少为 1
        if x_max <= x_min:
            x_max = x_min + 1
        if y_max <= y_min:
            y_max = y_min + 1
        
        return (x_min, y_min, x_max, y_max)
    
    def _crop_and_resize(self, frame, bbox, target_size):
        """从 frame 裁剪 bbox 区域并调整到 target_size"""
        x1, y1, x2, y2 = bbox
        
        # 裁剪 ROI
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            # 返回黑色图像
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # 调整大小
        roi_resized = cv2.resize(roi, (target_size, target_size))
        
        return roi_resized
    
    def _random_crop_88(self, roi_96, offset=None):
        """
        从 96x96 随机裁剪到 88x88（训练用）
        
        Args:
            roi_96: 96x96 输入图像
            offset: 可选的固定偏移量 (top, left)，如果提供则使用该偏移量
        
        Returns:
            88x88 裁剪图像
        """
        if offset is not None:
            top, left = offset
        else:
            top = np.random.randint(0, 96 - 88 + 1)
            left = np.random.randint(0, 96 - 88 + 1)
        
        roi_88 = roi_96[top:top+88, left:left+88]
        return roi_88
    
    def _center_crop_88(self, roi_96):
        """从 96x96 中心裁剪到 88x88（验证/测试用）"""
        top = (96 - 88) // 2
        left = (96 - 88) // 2
        roi_88 = roi_96[top:top+88, left:left+88]
        return roi_88
    
    def process_video(self, input_video_path, output_dir, mode='train'):
        """
        处理单个视频文件，生成所有输出
        
        Args:
            input_video_path: 输入视频路径
            output_dir: 输出目录
            mode: 'train' 或 'eval'（决定 88x88 的裁剪方式）
        
        Returns:
            dict: 输出文件路径字典
        """
        # 创建视频输出目录（以视频名为子目录）
        video_name = Path(input_video_path).stem
        video_output_dir = Path(output_dir) / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 打开输入视频
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 初始化输出视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        outputs = {
            'mouth_96': cv2.VideoWriter(str(video_output_dir / 'mouth_96.mp4'), fourcc, fps, (96, 96)),
            'face_96': cv2.VideoWriter(str(video_output_dir / 'face_96.mp4'), fourcc, fps, (96, 96)),
            'mouth_88': cv2.VideoWriter(str(video_output_dir / 'mouth_88.mp4'), fourcc, fps, (88, 88)),
            'face_88': cv2.VideoWriter(str(video_output_dir / 'face_88.mp4'), fourcc, fps, (88, 88)),
            'overlay_debug': cv2.VideoWriter(str(video_output_dir / 'overlay_debug.mp4'), fourcc, fps, (width, height))
        }
        
        # 重置平滑状态
        self.smoothed_mouth_bbox = None
        self.smoothed_face_bbox = None
        
        # 如果是 train 模式，为整个视频生成固定的随机 offset
        # 这样同一个视频的所有帧使用相同的裁剪位置，避免帧间抖动
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
            
            # 转换为 RGB 供 MediaPipe 处理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            mouth_bbox = None
            face_bbox = None
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # 获取嘴部 bbox
                mouth_bbox = self._get_mouth_bbox(face_landmarks, width, height)
                
                # 获取全脸 bbox
                face_bbox = self._get_face_bbox(face_landmarks, width, height)
            
            # 平滑或回填 mouth bbox
            if mouth_bbox is not None:
                mouth_bbox_smoothed = self._smooth_bbox(mouth_bbox, 'mouth')
            elif self.smoothed_mouth_bbox is not None:
                # 回填上一帧
                cx, cy, w, h = self.smoothed_mouth_bbox
                mouth_bbox_smoothed = (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))
            else:
                # 首帧就检测失败，使用默认框（画面中央）
                cx, cy = width // 2, height // 2
                default_size = min(width, height) // 4
                mouth_bbox_smoothed = (cx - default_size//2, cy - default_size//2, 
                                       cx + default_size//2, cy + default_size//2)
            
            # 平滑或回填 face bbox
            if face_bbox is not None:
                face_bbox_smoothed = self._smooth_bbox(face_bbox, 'face')
            elif self.smoothed_face_bbox is not None:
                # 回填上一帧
                cx, cy, w, h = self.smoothed_face_bbox
                face_bbox_smoothed = (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))
            else:
                # 首帧就检测失败，使用默认框
                cx, cy = width // 2, height // 2
                default_size = int(min(width, height) * 0.8)
                face_bbox_smoothed = (cx - default_size//2, cy - default_size//2,
                                      cx + default_size//2, cy + default_size//2)
            
            # 裁剪到图像边界
            mouth_bbox_clipped = self._clip_bbox(mouth_bbox_smoothed, width, height)
            face_bbox_clipped = self._clip_bbox(face_bbox_smoothed, width, height)
            
            # 生成 mouth_96 和 face_96
            mouth_96 = self._crop_and_resize(frame, mouth_bbox_clipped, 96)
            face_96 = self._crop_and_resize(frame, face_bbox_clipped, 96)
            
            # 生成 mouth_88 和 face_88
            if mode == 'train':
                # 使用固定的 offset，确保同一视频所有帧裁剪位置一致
                mouth_88 = self._random_crop_88(mouth_96, offset=fixed_crop_offset)
                face_88 = self._random_crop_88(face_96, offset=fixed_crop_offset)
            else:
                mouth_88 = self._center_crop_88(mouth_96)
                face_88 = self._center_crop_88(face_96)
            
            # 写入输出视频
            outputs['mouth_96'].write(mouth_96)
            outputs['face_96'].write(face_96)
            outputs['mouth_88'].write(mouth_88)
            outputs['face_88'].write(face_88)
            
            # 生成 overlay_debug
            overlay_frame = frame.copy()
            
            # 画 mouth bbox（绿色）
            x1, y1, x2, y2 = mouth_bbox_clipped
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay_frame, 'Mouth', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 画 face bbox（蓝色）
            x1, y1, x2, y2 = face_bbox_clipped
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(overlay_frame, 'Face', (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 添加帧号和时间戳
            time_sec = frame_idx / fps if fps > 0 else 0
            text = f'Frame: {frame_idx} | Time: {time_sec:.2f}s | FPS: {fps}'
            cv2.putText(overlay_frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            outputs['overlay_debug'].write(overlay_frame)
            
            frame_idx += 1
        
        # 释放资源
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
        批量处理目录中的所有视频
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            mode: 'train' 或 'eval'
            pattern: 文件匹配模式（默认递归搜索所有 mp4）
        
        Returns:
            list: 所有输出文件路径列表
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


# 使用示例
if __name__ == '__main__':
    extractor = ROIsExtractor()
    
    # 处理单个文件
    # outputs = extractor.process_video('path/to/video.mp4', './output', mode='train')
    
    # 批量处理目录
    # all_outputs = extractor.process_directory('path/to/dataset', './output', mode='train')
    
    print("ROI Extractor ready!")
    print("Usage:")
    print("  extractor.process_video('video.mp4', './output', mode='train')")
    print("  extractor.process_directory('./videos', './output', mode='eval')")
