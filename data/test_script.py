# Zhenning
import cv2
import numpy as np
from roi_extractor import ROIsExtractor
import os
import glob
from pathlib import Path


def collect_video_paths(root_dir, pattern='**/*.mp4'):
    """
    Collect all video file paths under the directory
    
    Args:
        root_dir: root directory
        pattern: file matching pattern (default: recursively search all mp4 files)
    
    Returns:
        list: list of video file paths
    """
    video_paths = glob.glob(os.path.join(root_dir, pattern), recursive=True)
    print(f"Found {len(video_paths)} videos in {root_dir}")
    return video_paths


def validate_outputs(video_name, output_dir, expected_fps, expected_frames, fps_tolerance=1.0):
    """
    Validate whether the output files are generated correctly
    
    Args:
        video_name: video name
        output_dir: output directory
        expected_fps: expected frame rate
        expected_frames: expected frame count
        fps_tolerance: allowed fps error range (default: ±1.0 fps)
    
    Returns:
        dict: validation results
    """
    video_output_dir = Path(output_dir) / video_name
    
    required_files = [
        'mouth_96.mp4',
        'face_96.mp4',
        'mouth_88.mp4',
        'face_88.mp4',
        'overlay_debug.mp4'
    ]
    
    results = {
        'all_files_exist': True,
        'files': {},
        'fps_match': True,
        'frame_count_match': True,
        'issues': []
    }
    
    for filename in required_files:
        filepath = video_output_dir / filename
        if not filepath.exists():
            results['all_files_exist'] = False
            results['issues'].append(f"Missing file: {filename}")
            continue
        
        # Check video properties
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            results['issues'].append(f"Cannot open file: {filename}")
            cap.release()
            continue
        
        # Read original fps (float) to avoid precision loss
        fps_float = cap.get(cv2.CAP_PROP_FPS)
        fps = int(round(fps_float))  # Round to nearest integer
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        results['files'][filename] = {
            'fps': fps,
            'fps_float': fps_float,
            'frame_count': frame_count
        }
        
        # Validate fps (use tolerance comparison to avoid floating-point errors)
        # Allow ±fps_tolerance difference, or relative error < 1%
        fps_diff = abs(fps_float - expected_fps)
        fps_relative_error = fps_diff / expected_fps if expected_fps > 0 else 0
        
        if fps_diff > fps_tolerance and fps_relative_error > 0.01:
            results['fps_match'] = False
            results['issues'].append(
                f"{filename}: FPS mismatch ({fps_float:.2f} vs {expected_fps}, diff={fps_diff:.2f})"
            )
        
        # Validate frame count (allow ±1 difference, since some encoders may have slight variation)
        if abs(frame_count - expected_frames) > 1:
            results['frame_count_match'] = False
            results['issues'].append(
                f"{filename}: Frame count mismatch ({frame_count} vs {expected_frames})"
            )
    
    return results


def validate_on_dataset(dataset_name, video_paths, output_dir, sample_size=10, mode='eval'):
    """
    Validate the ROI extractor on a dataset
    
    Args:
        dataset_name: dataset name
        video_paths: list of video paths
        output_dir: output directory
        sample_size: number of samples
        mode: 'train' or 'eval'
    
    Returns:
        dict: validation metrics
    """
    if not video_paths:
        print(f"No videos found for {dataset_name}")
        return None
    
    extractor = ROIsExtractor()
    
    metrics = {
        'total_videos': len(video_paths),
        'processed_videos': 0,
        'failed_videos': 0,
        'total_frames': 0,
        'success_rate': 0.0,
        'avg_bbox_size': 0.0,
        'validation_results': []
    }
    
    # Randomly sample videos for testing
    if len(video_paths) > sample_size:
        test_videos = np.random.choice(video_paths, sample_size, replace=False)
    else:
        test_videos = video_paths
    
    print(f"\n{'='*60}")
    print(f"Validating {dataset_name} Dataset")
    print(f"Sample size: {len(test_videos)} videos")
    print(f"{'='*60}")
    
    for video_idx, video_path in enumerate(test_videos, 1):
        video_name = Path(video_path).stem
        print(f"\n[{video_idx}/{len(test_videos)}] Processing: {video_name}")
        
        # Get input video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open video: {video_path}")
            metrics['failed_videos'] += 1
            continue
        
        input_fps = int(cap.get(cv2.CAP_PROP_FPS))
        input_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Process video
        try:
            outputs = extractor.process_video(video_path, output_dir, mode=mode)
            if not outputs:
                print(f"  [ERROR] Processing failed")
                metrics['failed_videos'] += 1
                continue
            
            metrics['processed_videos'] += 1
            metrics['total_frames'] += input_frame_count
            
            # Validate output files
            validation = validate_outputs(video_name, output_dir, input_fps, input_frame_count)
            metrics['validation_results'].append(validation)
            
            # Print results
            if validation['all_files_exist'] and validation['fps_match'] and validation['frame_count_match']:
                print(f"  [OK] All outputs validated successfully")
            else:
                print(f"  [WARNING] Validation issues:")
                for issue in validation['issues']:
                    print(f"    - {issue}")
            
        except Exception as e:
            print(f"  [ERROR] Exception: {e}")
            metrics['failed_videos'] += 1
    
    # Calculate overall metrics
    metrics['success_rate'] = metrics['processed_videos'] / len(test_videos) if len(test_videos) > 0 else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"{dataset_name} Validation Summary")
    print(f"{'='*60}")
    print(f"Total videos sampled: {len(test_videos)}")
    print(f"Successfully processed: {metrics['processed_videos']}")
    print(f"Failed: {metrics['failed_videos']}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Total frames processed: {metrics['total_frames']}")
    
    # Count validation issues
    all_issues = []
    for result in metrics['validation_results']:
        all_issues.extend(result['issues'])
    
    if all_issues:
        print(f"\nIssues found: {len(all_issues)}")
        # Show the first 10 issues
        for issue in all_issues[:10]:
            print(f"  - {issue}")
        if len(all_issues) > 10:
            print(f"  ... and {len(all_issues) - 10} more issues")
    
    return metrics


def quick_preview(output_dir, video_name, num_frames=3):
    """
    Quickly preview the output results
    
    Args:
        output_dir: output directory
        video_name: video name
        num_frames: number of preview frames
    """
    video_output_dir = Path(output_dir) / video_name
    
    print(f"\n{'='*60}")
    print(f"Quick Preview: {video_name}")
    print(f"{'='*60}")
    
    # Check whether files exist
    files = {
        'Mouth 96x96': video_output_dir / 'mouth_96.mp4',
        'Face 96x96': video_output_dir / 'face_96.mp4',
        'Overlay Debug': video_output_dir / 'overlay_debug.mp4'
    }
    
    for name, filepath in files.items():
        if not filepath.exists():
            print(f"{name}: [NOT FOUND]")
            continue
        
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            print(f"{name}: [CANNOT OPEN]")
            cap.release()
            continue
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{name}:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {frame_count}")
        
        # Sample a few frames for checking
        sample_indices = np.linspace(0, frame_count - 1, min(num_frames, frame_count), dtype=int)
        frames_valid = []
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            frames_valid.append(ret)
        
        print(f"  Sampled frames valid: {sum(frames_valid)}/{len(frames_valid)}")
        
        cap.release()


if __name__ == '__main__':
    # Configure paths
    grid_root = "./grid_video_path"
    voxceleb_root = "./VoxCeleb_video_path"
    output_dir = "./roi_output"
    
    # Collect video paths
    print("Collecting video paths...")
    grid_videos = collect_video_paths(grid_root, pattern='**/*.mp4')
    voxceleb_videos = collect_video_paths(voxceleb_root, pattern='**/*.mp4')
    
    # Validate GRID dataset
    if grid_videos:
        grid_metrics = validate_on_dataset(
            "GRID", 
            grid_videos, 
            output_dir, 
            sample_size=5,  # Test a small number of samples first
            mode='eval'
        )
        
        # Quickly preview the first successfully processed video
        if grid_metrics and grid_metrics['processed_videos'] > 0:
            # Find the first successful video
            video_outputs_dir = Path(output_dir)
            first_video_dir = sorted(video_outputs_dir.iterdir())[0]
            quick_preview(output_dir, first_video_dir.name)
    
    # Validate VoxCeleb dataset
    if voxceleb_videos:
        voxceleb_metrics = validate_on_dataset(
            "VoxCeleb", 
            voxceleb_videos, 
            output_dir, 
            sample_size=5,
            mode='eval'
        )
    
    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nExpected output structure for each video:")
    print("  roi_output/")
    print("    video_name/")
    print("      - mouth_96.mp4")
    print("      - face_96.mp4")
    print("      - mouth_88.mp4")
    print("      - face_88.mp4")
    print("      - overlay_debug.mp4")

