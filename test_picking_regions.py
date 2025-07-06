import cv2
import numpy as np
import os
import argparse

# Assuming these classes are available in the text_frame_extractor package
# We'll need to adjust imports if this script is not run from the root directory
from text_frame_extractor.region_detection import RegionDetector

def process_video_frames(video_path: str, output_dir: str, max_frames: int = -1):
    """
    Processes a video frame by frame, extracts the detected region,
    and saves it on a white background.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    region_detector = RegionDetector()
    # FrameAligner is not strictly needed for this task as we are just copying
    # the region, not aligning it to a fixed output size.
    # However, if future requirements involve resizing the extracted region,
    # FrameAligner would be useful.

    frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames != -1 and frame_count >= max_frames):
            break

        height, width, _ = frame.shape
        
        # Create a white canvas
        white_canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        # Detect the region of interest
        regions = region_detector.detect(frame)

        for region in regions:
            poly = np.asarray(region.polygon, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(poly)
            detected_region = frame[y : y + h, x : x + w]
            white_canvas[y : y + h, x : x + w] = detected_region
            cv2.polylines(white_canvas, [poly], True, (0, 255, 0), 2)
            print(f"Frame {frame_count} region score: {region.score:.2f}")

        # Construct output filename
        output_filename = os.path.join(output_dir, f"{video_name}_{frame_count:04d}.jpg")
        
        # Save the processed frame
        cv2.imwrite(output_filename, white_canvas)
        
        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames from {video_path} and saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts and processes frames from a video.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the processed frames.")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum number of frames to process. -1 for no limit.")
    args = parser.parse_args()

    process_video_frames(args.video_path, args.output_dir, args.max_frames)
