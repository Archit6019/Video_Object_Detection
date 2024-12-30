# PaliGemma Object Detection

This project uses the PaliGemma model to detect specified objects in images or videos and draw bounding boxes around them.

## Features

- Detect objects in images or videos using the PaliGemma model.
- Supports input from URLs, local image files, and video files.
- Outputs processed images or videos with bounding boxes.

## Requirements

- Python 3.7 or higher
- CUDA-enabled GPU (optional for faster processing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/paligemma-object-detection.git
   cd paligemma-object-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To detect objects in an image or video, run the following command:
```bash
python main.py <input_path> <object_to_detect> [-o <output_path>]
```
  - `<input_path>`: URL, path to the image file, or path to the video file.
  - `<object_to_detect>`: The object you want to detect.
  - `-o <output_path>`: (Optional) Path to save the output image or video.

## Example 

```bash
python main.py example.jpg "cat" -o output.jpg

```
