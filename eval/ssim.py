import argparse

import cv2
# Import the specific metric function from scikit-image
from skimage.metrics import structural_similarity


def main():
    """
    Main function to load two videos and calculate the average SSIM between them
    using the scikit-image library.
    """
    parser = argparse.ArgumentParser(
        description="Compute SSIM between two video files.")
    parser.add_argument("--ref-video", required=True, help="Reference video path.")
    parser.add_argument("--gen-video", required=True, help="Generated video path.")
    args = parser.parse_args()

    path_to_video1 = args.ref_video
    path_to_video2 = args.gen_video
    print(f"Loading original video: {path_to_video1}")
    print(f"Loading compressed video: {path_to_video2}")

    # Open the video files
    cap1 = cv2.VideoCapture(path_to_video1)
    cap2 = cv2.VideoCapture(path_to_video2)

    # Check if videos opened successfully
    if not cap1.isOpened():
        print(f"Error: Could not open original video at '{path_to_video1}'")
        return
    if not cap2.isOpened():
        print(f"Error: Could not open compressed video at '{path_to_video2}'")
        cap1.release()  # Release the first capture if the second one fails
        return

    # Check for consistent properties
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count1 != frame_count2:
        print(
            f"\nWarning: Videos have a different number of frames ({frame_count1} vs {frame_count2})."
        )
        print("Comparison will stop when the shorter video ends.\n")

    total_ssim = 0.0
    frame_number = 0

    while True:
        # Read one frame from each video
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # If either video has ended, break the loop
        if not ret1 or not ret2:
            break

        # Ensure frames have the same dimensions before comparison
        if frame1.shape != frame2.shape:
            print(
                f"Error on frame {frame_number + 1}: Frame dimensions do not match."
            )
            print(
                f"Video 1 frame shape: {frame1.shape}, Video 2 frame shape: {frame2.shape}"
            )
            # Optionally, resize one frame to match the other, or just skip
            continue  # Skip this frame pair

        frame_number += 1

        # --- SSIM CALCULATION ---
        # The 'structural_similarity' function requires the 'channel_axis' parameter
        # for multi-channel (color) images. OpenCV loads images in BGR format, so
        # the shape is (height, width, channels). The channel axis is the last one (index 2).
        # We also specify the data_range, which is 255 for uint8 images.
        ssim_value = structural_similarity(frame1,
                                           frame2,
                                           channel_axis=2,
                                           data_range=255)

        total_ssim += ssim_value
        # print(f"Frame {frame_number}: SSIM = {ssim_value:.6f}")

    # --- Summary ---
    print("\n--- Processing Complete ---")
    if frame_number > 0:
        average_ssim = total_ssim / frame_number
        print(
            f"Average SSIM (across {frame_number} frames): {average_ssim:.6f}")
    else:
        print("No frames were processed. Check video paths and integrity.")

    # Release the video capture objects
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
