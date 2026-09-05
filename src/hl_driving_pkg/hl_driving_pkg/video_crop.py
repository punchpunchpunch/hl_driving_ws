import cv2
import subprocess
import os
import sys

# ============================================================/WIN_20260905_15_57_18_Pro
# Paths
# ============================================================src/hl_driving_pkg/hl_driving_pkg/vision_dataset/WIN_20260905_15_57_18_Pro.mp4
input_path = "/home/yeong/hl_driving_ws/src/hl_driving_pkg/hl_driving_pkg/vision_dataset/WIN_20260905_15_58_25_Pro.mp4"

output_path = "/home/yeong/hl_driving_ws/src/hl_driving_pkg/hl_driving_pkg/vision_dataset/WIN_20260905_15_58_25_Pro_crop.mp4"

# Temporary file
temp_path = "/tmp/cropped_temp.mp4"

# ============================================================
# Target size
# ============================================================
TARGET_WIDTH = 1280
TARGET_HEIGHT = 400

# ============================================================
# Open input video
# ============================================================
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("ERROR: Cannot open input video")
    print(input_path)
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Input video : {width} x {height}")
print(f"FPS         : {fps:.3f}")

# Check resolution
if width < TARGET_WIDTH or height < TARGET_HEIGHT:
    print("ERROR: Input video is smaller than 1280x400.")
    cap.release()
    sys.exit(1)

# ============================================================
# Temporary crop video
# ============================================================
# mp4v is only used temporarily.
# It will be converted to H.264 by FFmpeg afterward.
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    temp_path,
    fourcc,
    fps,
    (TARGET_WIDTH, TARGET_HEIGHT)
)

if not out.isOpened():
    print("ERROR: Cannot create temporary video.")
    cap.release()
    sys.exit(1)

# ============================================================
# Crop
# Top-left 기준: 1280 x 400
# ============================================================
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Top 400 pixels
    cropped = frame[0:TARGET_HEIGHT, 0:TARGET_WIDTH]

    out.write(cropped)

    frame_count += 1

    if frame_count % 100 == 0:
        print(f"Processed frames: {frame_count}")

cap.release()
out.release()

print(f"\nCrop complete: {frame_count} frames")

# ============================================================
# Convert to H.264 using FFmpeg
# ============================================================
print("Converting to H.264...")

command = [
    "ffmpeg",
    "-y",

    "-i", temp_path,

    # H.264 encoder
    "-c:v", "libx264",

    # Good compatibility
    "-preset", "medium",
    "-crf", "23",
    "-pix_fmt", "yuv420p",

    # Make MP4 browser-friendly
    "-movflags", "+faststart",

    # No audio
    "-an",

    output_path
]

result = subprocess.run(command)

# ============================================================
# Check result
# ============================================================
if result.returncode != 0:
    print("ERROR: FFmpeg conversion failed.")
    sys.exit(1)

# Delete temporary file
if os.path.exists(temp_path):
    os.remove(temp_path)

print("\n========================================")
print("DONE")
print("========================================")
print(f"Output: {output_path}")
print(f"Resolution: {TARGET_WIDTH} x {TARGET_HEIGHT}")
print("Codec: H.264")
print("Pixel format: yuv420p")
print("========================================")