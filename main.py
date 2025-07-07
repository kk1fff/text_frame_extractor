import argparse
import cv2
from text_frame_extractor import process_video


def main():
    parser = argparse.ArgumentParser(description="Process video and reconstruct text region")
    parser.add_argument("input_video", help="Path to input video")
    parser.add_argument("output_image", help="Path to save reconstructed image")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save intermediate results")
    parser.add_argument("--debug-output-dir", default="local.debug", help="Directory to save debug output (default: local.debug)")
    args = parser.parse_args()

    image, text, score = process_video(args.input_video, debug_mode=args.debug, debug_output_dir=args.debug_output_dir)
    cv2.imwrite(args.output_image, image)
    print("Extracted Text:\n", text)
    print("Quality Score:", score)
    
    if args.debug:
        print(f"Debug output saved to: {args.debug_output_dir}")


if __name__ == "__main__":
    main()
