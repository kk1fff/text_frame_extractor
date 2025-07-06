import argparse
import cv2
from text_frame_extractor import process_video


def main():
    parser = argparse.ArgumentParser(description="Process video and reconstruct text region")
    parser.add_argument("input_video", help="Path to input video")
    parser.add_argument("output_image", help="Path to save reconstructed image")
    args = parser.parse_args()

    image, text, score = process_video(args.input_video)
    cv2.imwrite(args.output_image, image)
    print("Extracted Text:\n", text)
    print("Quality Score:", score)


if __name__ == "__main__":
    main()
