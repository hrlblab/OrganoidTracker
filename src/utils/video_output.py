#!/usr/bin/env python3
"""
Video Output Utilities for Medical-SAM2
Handles creation of different video output types with robust codec support
"""

import cv2
import numpy as np
import torch
from pathlib import Path


class VideoOutputGenerator:
    """
    Generate different types of video outputs from tracking results
    """

    def __init__(self):
        self.supported_types = ['original', 'overlay', 'mask', 'side_by_side']
        self._codec_cache = {}  # Cache successful codecs
        self._system_codecs_tested = False
        self._debug_session_timestamp = None  # For organizing debug frames by session

    def reset_debug_session(self):
        """Reset debug session for new video processing"""
        self._debug_session_timestamp = None

    def _detect_system_codecs(self):
        """Detect which codecs are available on the system"""
        if self._system_codecs_tested:
            return

        print("🔍 Detecting available video codecs...")
        test_codecs = [
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
            ('I420', cv2.VideoWriter_fourcc(*'I420')),
            ('IYUV', cv2.VideoWriter_fourcc(*'IYUV')),
        ]

        # Test with small dummy video
        test_path = "temp_codec_test.mp4"
        for codec_name, fourcc in test_codecs:
            try:
                writer = cv2.VideoWriter(test_path, fourcc, 30, (100, 100))
                if writer.isOpened():
                    self._codec_cache[codec_name] = fourcc
                    print(f"   ✅ {codec_name} codec available")
                writer.release()
            except:
                print(f"   ❌ {codec_name} codec not available")

        # Clean up test file
        try:
            import os
            if os.path.exists(test_path):
                os.remove(test_path)
        except:
            pass

        self._system_codecs_tested = True

    def _optimize_fps_for_resolution(self, fps, width, height):
        """Optimize FPS based on video resolution for better performance"""
        max_dimension = max(width, height)

        # For very large videos, reduce FPS to improve encoding speed
        if max_dimension > 3000:
            optimized_fps = min(fps, 10.0)  # Max 10 FPS for 4K+
        elif max_dimension > 2000:
            optimized_fps = min(fps, 15.0)  # Max 15 FPS for 2K+
        elif max_dimension > 1000:
            optimized_fps = min(fps, 24.0)  # Max 24 FPS for 1080p+
        else:
            optimized_fps = fps  # Keep original FPS for smaller videos

        if optimized_fps < fps:
            print(f"   Optimizing FPS: {fps:.1f} → {optimized_fps:.1f} for {width}x{height} video")

        return optimized_fps

    def create_video(self, frames, video_segments, obj_id, output_path,
                    fps=30.0, video_type='overlay', color=(255, 0, 0), alpha=0.3):
        """
        Create MP4 video from frames and segmentation results

        Args:
            frames: List of original video frames (RGB format)
            video_segments: Tracking results from MedicalSAM2Tracker
            obj_id: Object ID to visualize
            output_path: Path for output video file
            fps: Output frame rate
            video_type: Type of video ('original', 'overlay', 'mask', 'side_by_side')
            color: Overlay color for segmentation (RGB)
            alpha: Transparency for overlay

        Returns:
            str: Path to created video file
        """
        if video_type not in self.supported_types:
            raise ValueError(f"Unsupported video type: {video_type}. Use one of {self.supported_types}")

        print(f"🎬 Creating {video_type} video...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Detect available codecs on first use
        self._detect_system_codecs()

        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Optimize FPS for large videos
        optimized_fps = self._optimize_fps_for_resolution(fps, width, height)

        # Calculate target dimensions based on video type
        target_width, target_height = self._get_target_dimensions(width, height, video_type)

        # Initialize video writer with robust codec handling
        video_writer = self._create_video_writer(output_path, optimized_fps, target_width, target_height)

        try:
            # Generate frames
            for frame_idx in range(len(frames)):
                output_frame = self._generate_frame(
                    frames[frame_idx], frame_idx, video_segments, obj_id,
                    video_type, color, alpha, target_width, target_height
                )

                # Apply scaling if needed for performance
                if hasattr(self, '_current_scale_factor') and self._current_scale_factor < 1.0:
                    output_frame = cv2.resize(
                        output_frame.astype(np.uint8),
                        (self._target_width, self._target_height),
                        interpolation=cv2.INTER_LINEAR
                    )

                # Convert RGB to BGR for OpenCV
                output_frame_bgr = cv2.cvtColor(output_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                video_writer.write(output_frame_bgr)

            video_writer.release()
            print(f"✅ Video saved: {output_path}")
            return str(output_path)

        except Exception as e:
            video_writer.release()
            raise RuntimeError(f"Error creating video: {str(e)}")

    def create_multi_object_video(self, frames, video_segments, output_path,
                                 fps=30.0, video_type='overlay', alpha=0.3, progress_callback=None, quality_scale=1.0, tracker=None):
        """
        Create video with multiple objects, each with different colors

        Args:
            frames: List of original video frames (RGB format)
            video_segments: Tracking results from tracker
            output_path: Path for output video file
            fps: Output frame rate
            video_type: Type of video ('overlay', 'mask', 'side_by_side')
            alpha: Transparency for overlay
            progress_callback: Optional callback(current, total, message) for progress updates
            quality_scale: Scale factor for quality/performance trade-off
            tracker: Tracker instance to determine reverse state (required for reverse tracking)

        Returns:
            str: Path to created video file
        """
        # Reset debug session for new video processing
        if hasattr(self, 'debug_mode') and self.debug_mode:
            self.reset_debug_session()

        print(f"🎬 Creating multi-object {video_type} video...")
        print(f"🔍 ENTRY DEBUG: frames={len(frames)}, video_segments={len(video_segments) if video_segments else 0}, tracker={tracker}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL: Check if we have any data to work with
        if not frames:
            print(f"❌ CRITICAL ERROR: No frames provided!")
            return None
            
        if not video_segments:
            print(f"❌ CRITICAL ERROR: No video_segments provided!")
            return None

        # Object colors (RGB format) - expanded palette for 20+ objects
        object_colors = {
            0: (128, 128, 128), # Gray - Background
            1: (255, 0, 0),     # Red
            2: (0, 255, 0),     # Green
            3: (0, 0, 255),     # Blue
            4: (255, 255, 0),   # Yellow
            5: (255, 0, 255),   # Magenta
            6: (0, 255, 255),   # Cyan
            7: (255, 165, 0),   # Orange
            8: (128, 0, 128),   # Purple
            9: (255, 192, 203), # Pink
            10: (165, 42, 42),  # Brown
            11: (144, 238, 144), # Light Green
            12: (135, 206, 235), # Sky Blue
            13: (221, 160, 221), # Plum
            14: (240, 230, 140), # Khaki
            15: (255, 99, 71),  # Tomato
            16: (64, 224, 208), # Turquoise
            17: (238, 130, 238), # Violet
            18: (255, 182, 193), # Light Pink
            19: (152, 251, 152), # Pale Green
            20: (245, 222, 179), # Wheat
        }

        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Memory safety check for very large videos
        frame_size_mb = (height * width * 3) / (1024 * 1024)  # RGB frame size in MB
        total_memory_mb = frame_size_mb * len(frames)

        print(f"📊 Video dimensions: {width}x{height}, Frame size: {frame_size_mb:.1f}MB, Total: {total_memory_mb:.1f}MB")

        if frame_size_mb > 50:  # Frames larger than 50MB each
            print(f"⚠️ Large frame size detected ({frame_size_mb:.1f}MB). Using aggressive scaling.")

        if total_memory_mb > 1000:  # Total video memory > 1GB
            print(f"⚠️ Large video memory footprint ({total_memory_mb:.1f}MB). Consider using fewer frames or lower resolution.")

        # Detect available codecs on first use
        self._detect_system_codecs()

        # Optimize FPS for large videos
        optimized_fps = self._optimize_fps_for_resolution(fps, width, height)

        target_width, target_height = self._get_target_dimensions(width, height, video_type, quality_scale)

        # Initialize video writer
        video_writer = self._create_video_writer(output_path, optimized_fps, target_width, target_height)

        try:
            # CONFIGURABLE PROCESSING: Check if video was reversed during tracking
            # Determine reverse state from tracker (if available) or use heuristic
            is_reversed_video = getattr(tracker, 'is_reversed_video', None)
            if is_reversed_video is None:
                # Heuristic: check if this looks like reversed processing
                # (later frames have tracking data but earlier ones don't)
                frame_indices = list(video_segments.keys()) if video_segments else []
                is_reversed_video = len(frame_indices) > 1 and max(frame_indices) > min(frame_indices) and frame_indices != sorted(frame_indices)
            
            print(f"🎬 Processing {len(frames)} frames ({'reverse-tracked' if is_reversed_video else 'forward-tracked'} video)")
            print(f"🔍 DEBUG: video_segments keys = {list(video_segments.keys()) if video_segments else 'None'}")
            print(f"🔍 DEBUG: is_reversed_video = {is_reversed_video}")
            
            # CRITICAL DEBUG: Check if video_segments has any actual mask data
            total_objects = 0
            frames_with_masks = 0
            for frame_idx, frame_objects in video_segments.items():
                if frame_objects:
                    frames_with_masks += 1
                    total_objects += len(frame_objects)
            print(f"🔍 CRITICAL: {frames_with_masks}/{len(video_segments)} frames have masks, {total_objects} total objects")

            processed_frames = []
            total_frames = len(frames)
            
            for frame_idx in range(total_frames):
                # Report progress for current frame
                if progress_callback:
                    progress_callback(frame_idx, total_frames, f"Processing frame {frame_idx + 1}/{total_frames}")

                # CONFIGURABLE MASK INDEXING: Use reversed index only if video was reverse-tracked
                if is_reversed_video:
                    # REVERSE TRACKING: Use reversed mask index to match reversed video frames
                    mask_idx = total_frames - 1 - frame_idx
                    frame_objects = list(video_segments[mask_idx].keys()) if mask_idx in video_segments else []
                else:
                    # FORWARD TRACKING: Use direct frame index (standard processing)
                    mask_idx = frame_idx
                    frame_objects = list(video_segments[mask_idx].keys()) if mask_idx in video_segments else []
                                # Conditional debug output based on debug_mode parameter
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"🔍 DEBUG: Frame {frame_idx} using mask {mask_idx}, has objects: {frame_objects}")

                    # ENHANCED DEBUG: Save debug frames for ALL objects
                    if mask_idx in video_segments and len(frame_objects) > 0:
                        # Save debug frames for all objects in this frame
                        self._save_debug_frames_for_all_objects(
                            frames[frame_idx], video_segments[mask_idx], frame_idx, mask_idx, video_type
                        )

                        # Print debug info for first object only (to avoid spam)
                        first_obj = frame_objects[0]
                        mask_data = video_segments[mask_idx][first_obj]

                        # Get mask content info
                        if hasattr(mask_data, 'cpu'):
                            mask_np = mask_data.cpu().numpy().squeeze()
                            non_zero_count = np.count_nonzero(mask_np > 0.5)
                            print(f"🔍 DEEP DEBUG: Frame {frame_idx} (mask {mask_idx}) obj {first_obj} mask shape: {mask_data.shape}, non-zero: {non_zero_count}")
                        elif hasattr(mask_data, 'shape'):
                            print(f"🔍 DEEP DEBUG: Frame {frame_idx} (mask {mask_idx}) obj {first_obj} mask shape: {mask_data.shape}")
                        print(f"🔍 DEEP DEBUG: Frame {frame_idx} (mask {mask_idx}) obj {first_obj} mask type: {type(mask_data)}")

                # Create custom video_segments dict with correct mask index
                frame_video_segments = {}
                if mask_idx in video_segments:
                    frame_video_segments[frame_idx] = video_segments[mask_idx]

                output_frame = self._generate_multi_object_frame(
                    frames[frame_idx], frame_idx, frame_video_segments, object_colors,
                    video_type, alpha, target_width, target_height
                )

                # Apply scaling if needed for performance
                if hasattr(self, '_current_scale_factor') and self._current_scale_factor < 1.0:
                    output_frame = cv2.resize(
                        output_frame.astype(np.uint8),
                        (self._target_width, self._target_height),
                        interpolation=cv2.INTER_LINEAR
                    )

                # Store processed frame for postprocessing
                processed_frames.append(output_frame)

                # Force garbage collection for large frames to prevent memory buildup
                if (frame_idx + 1) % 10 == 0:  # Every 10 frames
                    import gc
                    gc.collect()

            # CONFIGURABLE POSTPROCESSING: Only reverse frames if they were reverse-tracked
            if is_reversed_video:
                # REVERSE TRACKING: Reverse processed frames to restore original temporal order
                # Since SAM2 processed reversed input frames, we need to reverse the output 
                # to get back to original frame order
                processed_frames.reverse()
                
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"🔍 OUTPUT REVERSAL: Reversing processed frames to restore original temporal order")
                    print(f"🔍 DEBUG: After reversal, first frame shows original frame 0, last shows original frame {len(processed_frames)-1}")
            else:
                # FORWARD TRACKING: Frames are already in correct temporal order
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"🔍 NO REVERSAL NEEDED: Frames already in correct temporal order for forward tracking")

            # Write frames in correct temporal progression
            for i, output_frame in enumerate(processed_frames):
                # Report postprocessing progress
                if progress_callback:
                    progress_callback(total_frames + i, total_frames * 2, f"Writing frame {i + 1}/{total_frames}")

                # Convert RGB to BGR for OpenCV
                output_frame_bgr = cv2.cvtColor(output_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                video_writer.write(output_frame_bgr)

            # Report completion
            if progress_callback:
                progress_callback(total_frames * 2, total_frames * 2, f"Finalizing {video_type} video...")

            # Ensure proper VideoWriter cleanup
            video_writer.release()
            video_writer = None  # Clear reference

            # Force garbage collection after video completion
            import gc
            gc.collect()

            print(f"✅ Multi-object video saved: {output_path}")
            return str(output_path)

        except Exception as e:
            try:
                video_writer.release()
                video_writer = None
            except:
                pass
            # Force cleanup on error
            import gc
            gc.collect()
            raise RuntimeError(f"Error creating multi-object video: {str(e)}")

    def create_optimized_multi_object_videos(self, frames, video_segments, output_dir,
                                           fps=5.0, alpha=0.4, progress_callback=None, quality_scale=1.0, tracker=None):
        """
        🚀 OPTIMIZED: Create all video types with single mask processing pass

        This method processes masks once and reuses them for all video types,
        providing 2-6x speedup compared to the original approach.

        Args:
            frames: List of original video frames (RGB format)
            video_segments: Tracking results from tracker
            output_dir: Directory for output video files
            fps: Output frame rate
            alpha: Transparency for overlay
            progress_callback: Optional callback for progress updates
            quality_scale: Scale factor for quality/performance trade-off
            tracker: Tracker instance to determine reverse state

        Returns:
            dict: Paths to created videos {'overlay': path, 'mask': path, 'side_by_side': path}
        """
        from pathlib import Path
        import time

        output_dir = Path(output_dir)
        start_time = time.time()

        print(f"🚀 OPTIMIZATION: Single-pass mask processing for {len(frames)} frames")
        print(f"🔍 OPTIMIZED ENTRY DEBUG: frames={len(frames)}, video_segments={len(video_segments) if video_segments else 0}, tracker={tracker}")
        
        # CRITICAL: Check if we have any data to work with
        if not frames:
            print(f"❌ OPTIMIZED CRITICAL ERROR: No frames provided!")
            return {}
            
        if not video_segments:
            print(f"❌ OPTIMIZED CRITICAL ERROR: No video_segments provided!")
            return {}

        # Object colors (same as original)
        object_colors = {
            0: (128, 128, 128), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
            4: (255, 255, 0), 5: (255, 0, 255), 6: (0, 255, 255), 7: (255, 165, 0),
            8: (128, 0, 128), 9: (255, 192, 203), 10: (165, 42, 42), 11: (144, 238, 144),
            12: (135, 206, 235), 13: (221, 160, 221), 14: (240, 230, 140), 15: (255, 99, 71),
            16: (64, 224, 208), 17: (238, 130, 238), 18: (255, 182, 193), 19: (152, 251, 152),
            20: (245, 222, 179)
        }

        total_frames = len(frames)

        # STEP 1: Process all masks once and cache results
        if progress_callback:
            progress_callback(0, total_frames * 4, "🚀 Processing masks (optimization phase)")

        # Determine reverse state from tracker
        is_reversed_video = getattr(tracker, 'is_reversed_video', True)  # Default to True for backward compatibility
        
        # CRITICAL DEBUG: Check video_segments data
        total_objects = 0
        frames_with_masks = 0
        for frame_idx, frame_objects in video_segments.items():
            if frame_objects:
                frames_with_masks += 1
                total_objects += len(frame_objects)
        print(f"🔍 OPTIMIZED CRITICAL: {frames_with_masks}/{len(video_segments)} frames have masks, {total_objects} total objects")
        print(f"🔍 OPTIMIZED DEBUG: is_reversed_video = {is_reversed_video}")
        
        processed_frames = self._process_all_frames_optimized(
            frames, video_segments, object_colors, alpha, quality_scale, progress_callback, is_reversed_video
        )

        # STEP 2: Generate all video types from processed frames
        created_videos = {}
        video_types = ['overlay', 'mask', 'side_by_side']

        for i, video_type in enumerate(video_types):
            base_progress = total_frames + (i * total_frames)

            output_path = output_dir / f"multi_object_{video_type}.mp4"

            if progress_callback:
                progress_callback(base_progress, total_frames * 4, f"🎬 Creating {video_type} video (optimized)")

            try:
                print(f"🔍 CREATING {video_type} video from processed frames...")
                video_path = self._create_video_from_processed_frames(
                    processed_frames, video_type, str(output_path), fps, progress_callback, base_progress, is_reversed_video
                )
                created_videos[video_type] = video_path
                print(f"✅ {video_type} video created (optimized)")

            except Exception as e:
                print(f"❌ Error creating {video_type} video: {e}")
                import traceback
                traceback.print_exc()
                created_videos[video_type] = None

        optimization_time = time.time() - start_time

        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"🎉 OPTIMIZATION COMPLETE: All videos generated in {optimization_time:.2f}s")

        return created_videos

    def _process_all_frames_optimized(self, frames, video_segments, object_colors, alpha, quality_scale, progress_callback, is_reversed_video=True):
        """
        🚀 Process all frames once with all video type variants
        Returns: dict with all processed frame variants
        
        Args:
            is_reversed_video: Whether the tracking was done in reverse order
        """
        total_frames = len(frames)
        processed_frames = {
            'original': [],
            'overlay': [],
            'mask': [],
            'side_by_side': []
        }

        # Get frame dimensions and apply quality scaling
        height, width = frames[0].shape[:2]
        target_width = int(width * quality_scale)
        target_height = int(height * quality_scale)

        for frame_idx in range(total_frames):
            if progress_callback:
                progress_callback(frame_idx, total_frames * 4, f"Processing frame {frame_idx + 1}/{total_frames}")

            # CONFIGURABLE MASK INDEXING: Use reversed index only if video was reverse-tracked
            if is_reversed_video:
                # REVERSE TRACKING: Use reversed mask index to match reversed video frames
                mask_idx = (total_frames - 1) - frame_idx
            else:
                # FORWARD TRACKING: Use direct frame index (standard processing)
                mask_idx = frame_idx

            original_frame = frames[frame_idx]

            # Process original frame
            if original_frame.shape[:2] != (target_height, target_width):
                original_frame = cv2.resize(original_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # Initialize frame variants
            mask_frame = np.zeros_like(original_frame)
            overlay_frame = original_frame.copy().astype(np.float32)

            # Process all objects for this frame (SINGLE PASS!)
            objects_processed_this_frame = 0
            if mask_idx in video_segments:
                for obj_id, mask_data in video_segments[mask_idx].items():
                    if obj_id in object_colors:
                        # Process mask once per object per frame
                        mask = self._process_single_mask(mask_data, (target_height, target_width))
                        color = np.array(object_colors[obj_id])

                        if np.any(mask > 0):
                            objects_processed_this_frame += 1
                            # Update mask frame
                            for c in range(3):
                                mask_frame[:, :, c] = np.where(mask > 0, color[c], mask_frame[:, :, c])

                            # Update overlay frame
                            mask_3d = np.stack([mask] * 3, axis=-1)
                            overlay_frame = overlay_frame * (1 - mask_3d * alpha) + color * mask_3d * alpha
                        else:
                            if frame_idx == 0:  # Only debug first frame
                                print(f"🔍 MASK DEBUG: Frame {frame_idx} obj {obj_id} has empty mask")
                                
            # CRITICAL DEBUG: Report object processing for first few frames
            if frame_idx < 3:
                print(f"🔍 PROCESS DEBUG: Frame {frame_idx} -> mask_idx {mask_idx}, processed {objects_processed_this_frame} objects")

            # Finalize frame variants
            overlay_frame = overlay_frame.astype(np.uint8)
            side_by_side_frame = np.hstack([original_frame, overlay_frame])

            # Store all variants
            processed_frames['original'].append(original_frame)
            processed_frames['overlay'].append(overlay_frame)
            processed_frames['mask'].append(mask_frame)
            processed_frames['side_by_side'].append(side_by_side_frame)

        return processed_frames

    def _create_video_from_processed_frames(self, processed_frames, video_type, output_path, fps, progress_callback, base_progress, is_reversed_video=True):
        """
        Create video from pre-processed frames (NO mask processing!)
        
        Args:
            is_reversed_video: Whether the tracking was done in reverse order
        """
        frames = processed_frames[video_type]

        if not frames:
            raise ValueError(f"No processed frames available for {video_type}")

        # Get dimensions from first frame
        height, width = frames[0].shape[:2]

        # Use existing video writer creation method (handles codec selection)
        self._detect_system_codecs()
        video_writer = self._create_video_writer(output_path, fps, width, height)

        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
            
        # CRITICAL FIX: Use the actual scaled dimensions from video writer creation
        # The _create_video_writer method stores the scaled dimensions in instance variables
        actual_width = getattr(self, '_target_width', width)
        actual_height = getattr(self, '_target_height', height)
        print(f"🔧 Video writer created with actual dimensions: {actual_width}x{actual_height}")

        # CONFIGURABLE REVERSAL: Only reverse frames if they were reverse-tracked
        if is_reversed_video:
            # REVERSE TRACKING: Reverse frames to restore original temporal order (compensate for SAM2 preprocessing)
            frames_to_write = list(reversed(frames))
            print(f"🔄 Reversing {len(frames)} frames to restore original temporal order")
        else:
            # FORWARD TRACKING: Frames are already in correct order
            frames_to_write = frames
            print(f"▶️ Using {len(frames)} frames in original processing order")

        # Write frames (with proper scaling!)
        for i, frame in enumerate(frames_to_write):
            if progress_callback:
                progress_callback(base_progress + i, len(frames_to_write) * 4, f"Writing {video_type} frame {i + 1}")

            # CRITICAL FIX: Ensure frame matches video writer dimensions
            current_height, current_width = frame.shape[:2]
            if current_width != actual_width or current_height != actual_height:
                print(f"🔧 SCALING FRAME: {current_width}x{current_height} -> {actual_width}x{actual_height}")
                frame = cv2.resize(frame.astype(np.uint8), (actual_width, actual_height), interpolation=cv2.INTER_AREA)

            # CRITICAL DEBUG: Check frame content before writing
            if i == 0:  # Only debug first frame to avoid spam
                non_zero_pixels = np.count_nonzero(frame)
                frame_mean = np.mean(frame)
                print(f"🔍 FRAME DEBUG: First frame non-zero pixels={non_zero_pixels}, mean={frame_mean:.2f}, shape={frame.shape}")
                print(f"🔍 WRITER DEBUG: Video writer expects {actual_width}x{actual_height}")

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # CRITICAL DEBUG: Check if video writer is still valid
            if not video_writer.isOpened():
                print(f"❌ CRITICAL: Video writer closed unexpectedly at frame {i}")
                break
                
            video_writer.write(frame_bgr)

        video_writer.release()
        
        # CRITICAL DEBUG: Check final file size and validity
        import os
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"🔍 FILE DEBUG: Created {output_path} with size {file_size} bytes")
            if file_size <= 500:
                print(f"❌ CORRUPTION DETECTED: File size {file_size} bytes is suspiciously small!")
            else:
                print(f"✅ FILE LOOKS HEALTHY: {file_size} bytes")
        else:
            print(f"❌ CRITICAL: Output file {output_path} was not created!")
        
        return output_path

    def _get_target_dimensions(self, width, height, video_type, quality_scale=1.0):
        """Calculate target video dimensions with codec limit handling and quality scaling"""
        if video_type == 'side_by_side':
            target_width = width * 2
            target_height = height

            # Check codec limits
            max_dimension = 4096
            if target_width > max_dimension or target_height > max_dimension:
                scale_factor = min(max_dimension / target_width, max_dimension / target_height)
                target_width = int(target_width * scale_factor)
                target_height = int(target_height * scale_factor)
                print(f"   Scaling video to {target_width}x{target_height} for codec compatibility")
        else:
            target_width = width
            target_height = height

        # Apply quality scaling
        if quality_scale != 1.0:
            target_width = int(target_width * quality_scale)
            target_height = int(target_height * quality_scale)
            quality_names = {1.0: "original", 0.5: "mid", 0.25: "low"}
            quality_name = quality_names.get(quality_scale, f"{quality_scale:.2f}x")
            print(f"   Applying {quality_name} quality scaling: {width}x{height} → {target_width}x{target_height}")

        return target_width, target_height

    def _create_video_writer(self, output_path, fps, width, height):
        """Create video writer with optimized codec selection and performance tuning"""

        # Suppress OpenCV codec warnings
        import os
        original_opencv_log_level = os.environ.get('OPENCV_LOG_LEVEL', '')
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

        # Performance optimization: scale down very large videos for better encoding speed
        original_width, original_height = width, height
        scale_factor = 1.0

        # If video is very large (>2K), scale it down for faster encoding
        max_dimension = max(width, height)
        if max_dimension > 2048:
            scale_factor = 2048 / max_dimension
            width = int(width * scale_factor)
            height = int(height * scale_factor)
            # Ensure even dimensions for codec compatibility
            width = width if width % 2 == 0 else width - 1
            height = height if height % 2 == 0 else height - 1
            print(f"   Scaling video to {width}x{height} for better performance (scale: {scale_factor:.2f})")

        # Optimized codec selection based on resolution and system capabilities
        if max_dimension <= 1080:
            # For smaller videos, use high-quality codecs (prefer cached ones)
            preferred_codecs = ['mp4v', 'MJPG', 'XVID']
        else:
            # For large videos, prioritize speed and compatibility
            preferred_codecs = ['mp4v', 'MJPG', 'I420', 'IYUV']

        # Use cached codecs first, then fallback to testing
        codecs_to_try = []
        for codec_name in preferred_codecs:
            if codec_name in self._codec_cache:
                codecs_to_try.append((codec_name, self._codec_cache[codec_name]))

        # Add any remaining codecs not in cache
        all_codecs = [
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
            ('I420', cv2.VideoWriter_fourcc(*'I420')),
            ('IYUV', cv2.VideoWriter_fourcc(*'IYUV')),
        ]

        for codec_name, fourcc in all_codecs:
            if codec_name not in [c[0] for c in codecs_to_try]:
                codecs_to_try.append((codec_name, fourcc))

        # Store scale factor for frame processing
        self._current_scale_factor = scale_factor
        self._target_width = width
        self._target_height = height

        for codec_name, fourcc in codecs_to_try:
            try:
                # Test if codec works by creating a test writer
                test_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                if test_writer.isOpened():
                    print(f"   Using {codec_name} codec for {width}x{height} video")

                    # Restore original log level
                    if original_opencv_log_level:
                        os.environ['OPENCV_LOG_LEVEL'] = original_opencv_log_level
                    else:
                        os.environ.pop('OPENCV_LOG_LEVEL', None)

                    return test_writer
                else:
                    test_writer.release()

            except Exception as e:
                print(f"   {codec_name} codec failed: {e}")
                continue

        # If all optimized codecs fail, try basic fallback
        print("   Trying basic fallback codecs...")
        basic_codecs = [
            ('Raw', cv2.VideoWriter_fourcc(*'RGBA')),
            ('Uncompressed', 0),  # Uncompressed
        ]

        for codec_name, fourcc in basic_codecs:
            try:
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if writer.isOpened():
                    print(f"   Using {codec_name} codec (fallback)")
                    return writer
                else:
                    writer.release()
            except:
                continue

        # Restore log level before error
        if original_opencv_log_level:
            os.environ['OPENCV_LOG_LEVEL'] = original_opencv_log_level
        else:
            os.environ.pop('OPENCV_LOG_LEVEL', None)

        raise RuntimeError(f"Could not initialize video writer for {width}x{height} video with any codec. "
                          f"Original size: {original_width}x{original_height}")

    def _generate_frame(self, frame, frame_idx, video_segments, obj_id,
                       video_type, color, alpha, target_width, target_height):
        """Generate a single output frame based on video type"""

        if video_type == 'original':
            output_frame = frame.copy()

        elif video_type == 'mask':
            # Create mask-only frame
            mask = self._get_frame_mask(frame, frame_idx, video_segments, obj_id)
            # Convert to 3-channel image (white mask on black background)
            output_frame = np.stack([mask * 255] * 3, axis=-1)

            # Visual feedback for debugging
            if np.count_nonzero(mask) == 0:
                print(f"⚠️  No mask detected for frame {frame_idx}")
                # Create a red tinted frame to indicate no detection
                output_frame = frame.copy().astype(np.float32)
                output_frame[:, :, 0] = np.minimum(output_frame[:, :, 0] + 50, 255)  # Add red tint
                output_frame = output_frame.astype(np.uint8)

        elif video_type == 'overlay':
            # Create overlay with segmentation
            output_frame = frame.copy().astype(np.float32)
            mask = self._get_frame_mask(frame, frame_idx, video_segments, obj_id)

            if np.any(mask > 0):
                # Make overlay more visible
                mask_3d = np.stack([mask] * 3, axis=-1)
                output_frame = output_frame * (1 - mask_3d * alpha) + np.array(color) * mask_3d * alpha

            output_frame = output_frame.astype(np.uint8)

        elif video_type == 'side_by_side':
            # Create side-by-side comparison
            original_frame = frame.copy()
            overlay_frame = frame.copy().astype(np.float32)

            mask = self._get_frame_mask(frame, frame_idx, video_segments, obj_id)

            if np.any(mask > 0):
                # Make overlay more visible
                mask_3d = np.stack([mask] * 3, axis=-1)
                overlay_frame = overlay_frame * (1 - mask_3d * alpha) + np.array(color) * mask_3d * alpha

            overlay_frame = overlay_frame.astype(np.uint8)

            # Concatenate horizontally
            output_frame = np.hstack([original_frame, overlay_frame])

        # Resize if needed to match target dimensions
        current_height, current_width = output_frame.shape[:2]
        if current_width != target_width or current_height != target_height:
            output_frame = cv2.resize(output_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        return output_frame

    def _generate_multi_object_frame(self, frame, frame_idx, video_segments, object_colors,
                                   video_type, alpha, target_width, target_height):
        """Generate a single output frame with multiple objects"""

        if video_type == 'mask':
            # Create multi-object mask frame
            output_frame = np.zeros_like(frame)

            # Overlay each object with its color
            if frame_idx in video_segments:
                for obj_id, mask_data in video_segments[frame_idx].items():
                    if obj_id in object_colors:
                        mask = self._process_single_mask(mask_data, frame.shape[:2])
                        color = object_colors[obj_id]

                        # Apply color where mask is active
                        for c in range(3):
                            output_frame[:, :, c] = np.where(mask > 0, color[c], output_frame[:, :, c])

        elif video_type == 'overlay':
            # Create multi-object overlay
            output_frame = frame.copy().astype(np.float32)

            if frame_idx in video_segments:
                for obj_id, mask_data in video_segments[frame_idx].items():
                    if obj_id in object_colors:
                        mask = self._process_single_mask(mask_data, frame.shape[:2])
                        color = np.array(object_colors[obj_id])

                        if np.any(mask > 0):
                            # Apply colored overlay for this object
                            mask_3d = np.stack([mask] * 3, axis=-1)
                            output_frame = output_frame * (1 - mask_3d * alpha) + color * mask_3d * alpha

            output_frame = output_frame.astype(np.uint8)

        elif video_type == 'side_by_side':
            # Create side-by-side with multi-object overlay
            original_frame = frame.copy()
            overlay_frame = frame.copy().astype(np.float32)

            if frame_idx in video_segments:
                for obj_id, mask_data in video_segments[frame_idx].items():
                    if obj_id in object_colors:
                        mask = self._process_single_mask(mask_data, frame.shape[:2])
                        color = np.array(object_colors[obj_id])

                        if np.any(mask > 0):
                            mask_3d = np.stack([mask] * 3, axis=-1)
                            overlay_frame = overlay_frame * (1 - mask_3d * alpha) + color * mask_3d * alpha

            overlay_frame = overlay_frame.astype(np.uint8)
            output_frame = np.hstack([original_frame, overlay_frame])

        else:
            output_frame = frame.copy()

        # Resize if needed
        current_height, current_width = output_frame.shape[:2]
        if current_width != target_width or current_height != target_height:
            output_frame = cv2.resize(output_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        return output_frame

    def _process_single_mask(self, mask_data, frame_shape):
        """Process a single mask from tracking results"""
        # Handle tensor conversion
        if hasattr(mask_data, 'cpu'):
            mask = mask_data.cpu().numpy()
        elif hasattr(mask_data, 'numpy'):
            mask = mask_data.numpy()
        else:
            mask = mask_data

        if mask.ndim > 2:
            mask = mask.squeeze()

        # Convert logits to probabilities using sigmoid
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask)
        else:
            mask_tensor = mask

        mask_prob = torch.sigmoid(mask_tensor).numpy()
        mask_binary = (mask_prob > 0.5).astype(np.uint8)

        # Resize to match frame size if needed
        if mask_binary.shape != frame_shape:
            mask_binary = cv2.resize(
                mask_binary,
                (frame_shape[1], frame_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        return mask_binary

    def _get_frame_mask(self, frame, frame_idx, video_segments, obj_id):
        """Extract binary mask for a frame"""
        if frame_idx not in video_segments or obj_id not in video_segments[frame_idx]:
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Get mask from tracking results
        mask = video_segments[frame_idx][obj_id]

        # Handle tensor conversion
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        elif hasattr(mask, 'numpy'):
            mask = mask.numpy()

        if mask.ndim > 2:
            mask = mask.squeeze()

        # CRITICAL FIX: Convert logits to probabilities using sigmoid
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask)
        else:
            mask_tensor = mask

        # Apply sigmoid to convert logits to probabilities (0-1 range)
        mask_prob = torch.sigmoid(mask_tensor).numpy()

        # Convert to binary with proper threshold
        mask_binary = (mask_prob > 0.5).astype(np.uint8)

        # Resize to match frame size if needed
        if mask_binary.shape != frame.shape[:2]:
            mask_binary = cv2.resize(
                mask_binary,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        return mask_binary

    def _save_debug_frames_for_all_objects(self, frame, frame_objects_data, frame_idx, mask_idx, video_type):
        """Save debug frames for all objects in organized subfolders"""
        try:
            # Save debug frames for all video types when debug mode is enabled
            # (Note: Will create separate debug sessions for each video type)

            # Create main debug directory with persistent session timestamp
            if not hasattr(self, '_debug_session_timestamp') or self._debug_session_timestamp is None:
                from datetime import datetime
                self._debug_session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            debug_base_dir = Path("data/output_videos/debug_frames")
            debug_session_dir = debug_base_dir / f"session_{self._debug_session_timestamp}"
            debug_session_dir.mkdir(parents=True, exist_ok=True)

            # Save original frame once per frame (shared across all objects)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
            shared_frame_path = debug_session_dir / f"frame_{frame_idx:03d}_original.png"

            # Only save original frame once per frame
            if not shared_frame_path.exists():
                cv2.imwrite(str(shared_frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            # Process each object separately
            for obj_id, mask_data in frame_objects_data.items():
                # Create object-specific subdirectory
                obj_dir = debug_session_dir / f"object_{obj_id:02d}"
                obj_dir.mkdir(exist_ok=True)

                # Process mask data
                if hasattr(mask_data, 'cpu'):
                    mask_np = mask_data.cpu().numpy().squeeze()
                else:
                    mask_np = mask_data.squeeze() if hasattr(mask_data, 'squeeze') else mask_data

                # Convert logits to probabilities and then to 0-255
                if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
                    # Apply sigmoid to convert logits to probabilities
                    mask_prob = 1 / (1 + np.exp(-mask_np))  # sigmoid
                    mask_binary = (mask_prob > 0.5).astype(np.uint8) * 255
                else:
                    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

                # Resize mask to match frame if needed
                if mask_binary.shape != frame.shape[:2]:
                    mask_binary = cv2.resize(mask_binary, (frame.shape[1], frame.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

                # Save mask for this object
                mask_path = obj_dir / f"frame_{frame_idx:03d}_mask.png"
                cv2.imwrite(str(mask_path), mask_binary)

                # Create and save overlay for this object with unique color
                overlay = frame_rgb.copy().astype(np.float32)
                mask_3d = np.stack([mask_binary/255.0] * 3, axis=-1)

                # Use different colors for different objects
                colors = [
                    [255, 0, 0],    # Red
                    [0, 255, 0],    # Green
                    [0, 0, 255],    # Blue
                    [255, 255, 0],  # Yellow
                    [255, 0, 255],  # Magenta
                    [0, 255, 255],  # Cyan
                    [255, 128, 0],  # Orange
                    [128, 0, 255],  # Purple
                ]
                color = np.array(colors[(obj_id - 1) % len(colors)])

                alpha = 0.3
                overlay = overlay * (1 - mask_3d * alpha) + color * mask_3d * alpha
                overlay_path = obj_dir / f"frame_{frame_idx:03d}_overlay.png"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))

                # Save object-specific frame reference (symlink or copy)
                obj_frame_path = obj_dir / f"frame_{frame_idx:03d}_original.png"
                if not obj_frame_path.exists():
                    # Create a symlink to save space, fallback to copy if symlink fails
                    try:
                        obj_frame_path.symlink_to(f"../../frame_{frame_idx:03d}_original.png")
                    except:
                        cv2.imwrite(str(obj_frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            # Print debug info only once per session and create session summary
            if frame_idx == 0 and hasattr(self, 'debug_mode') and self.debug_mode:
                num_objects = len(frame_objects_data)
                print(f"💾 DEBUG: Saving organized debug frames to {debug_session_dir}")
                print(f"🔍 DEBUG: Processing {num_objects} objects with individual subfolders")
                print(f"📁 DEBUG: Structure: session_dir/object_XX/frame_YYY_[original|mask|overlay].png")

                # Create session info file
                self._create_debug_session_info(debug_session_dir, video_type, num_objects)

        except Exception as e:
            if frame_idx == 0:  # Only print once
                print(f"⚠️ Warning: Could not save debug frames for frame {frame_idx}: {e}")

    def _create_debug_session_info(self, debug_session_dir, video_type, num_objects):
        """Create a session info file with metadata about the debug session"""
        try:
            from datetime import datetime

            session_info = {
                'session_timestamp': self._debug_session_timestamp,
                'video_type': video_type,
                'num_objects': num_objects,
                'created_at': datetime.now().isoformat(),
                'debug_structure': {
                    'session_dir': f"session_{self._debug_session_timestamp}/",
                    'object_dirs': [f"object_{i:02d}/" for i in range(1, num_objects + 1)],
                    'file_types': ['frame_XXX_original.png', 'frame_XXX_mask.png', 'frame_XXX_overlay.png']
                },
                'description': 'Debug frames organized by object for kidney organoid cyst tracking analysis'
            }

            # Write JSON info file
            import json
            info_path = debug_session_dir / "session_info.json"
            with open(info_path, 'w') as f:
                json.dump(session_info, f, indent=2)

            # Write README file
            readme_path = debug_session_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(f"# Debug Session: {self._debug_session_timestamp}\n\n")
                f.write(f"**Video Type:** {video_type}\n")
                f.write(f"**Objects Tracked:** {num_objects}\n")
                f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Directory Structure\n\n")
                f.write("```\n")
                f.write(f"session_{self._debug_session_timestamp}/\n")
                f.write("├── session_info.json       # Session metadata\n")
                f.write("├── README.md               # This file\n")
                f.write("├── frame_XXX_original.png  # Original video frames\n")
                for i in range(1, num_objects + 1):
                    f.write(f"├── object_{i:02d}/\n")
                    f.write(f"│   ├── frame_XXX_original.png  # Original frame (symlink)\n")
                    f.write(f"│   ├── frame_XXX_mask.png      # Object mask\n")
                    f.write(f"│   └── frame_XXX_overlay.png   # Object overlay\n")
                f.write("```\n\n")
                f.write("## File Naming Convention\n\n")
                f.write("- `frame_XXX_original.png`: Original video frame (XXX = frame number, 3 digits)\n")
                f.write("- `frame_XXX_mask.png`: Binary mask for the object (white = object, black = background)\n")
                f.write("- `frame_XXX_overlay.png`: Original frame with colored object overlay\n\n")
                f.write("## Color Coding\n\n")
                f.write("Each object gets a unique color for easy identification:\n")
                f.write("- Object 1: Red\n- Object 2: Green\n- Object 3: Blue\n")
                f.write("- Object 4: Yellow\n- Object 5: Magenta\n- Object 6: Cyan\n")
                f.write("- Object 7: Orange\n- Object 8: Purple\n")
                f.write("- Additional objects cycle through colors\n")

        except Exception as e:
            print(f"⚠️ Warning: Could not create debug session info: {e}")

    def _save_debug_frame_and_mask(self, frame, mask_data, frame_idx, obj_id, video_type):
        """Legacy method - kept for compatibility"""
        # Convert to new method format
        frame_objects_data = {obj_id: mask_data}
        self._save_debug_frames_for_all_objects(frame, frame_objects_data, frame_idx, frame_idx, video_type)

    def create_multiple_videos(self, frames, video_segments, obj_id, output_dir,
                              fps=30.0, video_types=['overlay'], **kwargs):
        """
        Create multiple video types at once

        Args:
            frames: List of video frames
            video_segments: Tracking results
            obj_id: Object ID
            output_dir: Output directory
            fps: Frame rate
            video_types: List of video types to create
            **kwargs: Additional arguments for video creation

        Returns:
            dict: Mapping of video_type -> output_path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_videos = {}

        for video_type in video_types:
            try:
                output_path = output_dir / f"output_{video_type}.mp4"
                result_path = self.create_video(
                    frames, video_segments, obj_id, output_path,
                    fps=fps, video_type=video_type, **kwargs
                )
                created_videos[video_type] = result_path
            except Exception as e:
                print(f"❌ Error creating {video_type} video: {str(e)}")
                created_videos[video_type] = None

        return created_videos


def save_frame_sequence(frames, video_segments, obj_id, output_dir):
    """
    Save individual frames as PNG files

    Args:
        frames: List of video frames
        video_segments: Tracking results
        obj_id: Object ID
        output_dir: Output directory
    """
    from PIL import Image

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"

    for directory in [frames_dir, masks_dir, overlays_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    generator = VideoOutputGenerator()

    for frame_idx, frame in enumerate(frames):
        # Save original frame
        frame_pil = Image.fromarray(frame)
        frame_pil.save(frames_dir / f"frame_{frame_idx:04d}.png")

        # Save mask
        mask = generator._get_frame_mask(frame, frame_idx, video_segments, obj_id)
        mask_pil = Image.fromarray(mask * 255)
        mask_pil.save(masks_dir / f"mask_{frame_idx:04d}.png")

        # Save overlay
        overlay_frame = generator._generate_frame(
            frame, frame_idx, video_segments, obj_id,
            'overlay', (255, 0, 0), 0.3, frame.shape[1], frame.shape[0]
        )
        overlay_pil = Image.fromarray(overlay_frame.astype(np.uint8))
        overlay_pil.save(overlays_dir / f"overlay_{frame_idx:04d}.png")

    print(f"✅ Frame sequence saved to {output_dir}")


def create_summary_image(frames, video_segments, obj_id, output_path):
    """
    Create summary visualization showing key frames

    Args:
        frames: List of video frames
        video_segments: Tracking results
        obj_id: Object ID
        output_path: Path for summary image
    """
    import matplotlib.pyplot as plt

    num_frames = len(frames)

    # Select key frames
    key_frame_indices = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
    key_frame_indices = [idx for idx in key_frame_indices if idx < num_frames and idx in video_segments]

    if not key_frame_indices:
        print("⚠️  No frames available for summary")
        return

    # Create visualization
    fig, axes = plt.subplots(2, len(key_frame_indices), figsize=(4*len(key_frame_indices), 8))
    if len(key_frame_indices) == 1:
        axes = axes.reshape(2, 1)

    generator = VideoOutputGenerator()

    for i, frame_idx in enumerate(key_frame_indices):
        # Original frame
        axes[0, i].imshow(frames[frame_idx])
        axes[0, i].set_title(f'Frame {frame_idx}')
        axes[0, i].axis('off')

        # Frame with overlay
        overlay_frame = generator._generate_frame(
            frames[frame_idx], frame_idx, video_segments, obj_id,
            'overlay', (255, 0, 0), 0.3, frames[frame_idx].shape[1], frames[frame_idx].shape[0]
        )

        axes[1, i].imshow(overlay_frame.astype(np.uint8))
        axes[1, i].set_title(f'Segmentation {frame_idx}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Summary saved: {output_path}")