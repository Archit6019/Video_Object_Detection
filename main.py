from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image, ImageDraw
import requests
import torch 
import re 
import cv2
import argparse
import numpy as np

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype = torch.bfloat16,
    device_map="auto",
    revision="bfloat16",
).eval()
processor = AutoProcessor.from_pretrained(model_id)

def process_frame(frame, model , processor, object_to_detect):
    prompt = f"detect {object_to_detect}"
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    input_len = model_inputs['input_ids'].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

    detections = re.findall(r'<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>\s*(\w+)', decoded)

    height , width = frame.shape[:2]

    for detection in detections:
        y_min, x_min, y_max, x_max, label = detection
        y_min, x_min, y_max, x_max = [int(float(coord) / 1024 * (height if i % 2 == 0 else width)) 
                                      for i, coord in enumerate([y_min, x_min, y_max, x_max])]
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame

def detect_object(input_path: str, object_to_detect: str, output_path: str = None):
    if input_path.startswith(('http://', 'https://')):
        response = requests.get(input_path, stream=True)
        image = Image.open(response.raw)
        processed_image = process_frame(np.array(image), model, processor, object_to_detect)
        result = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        if output_path:
            result.save(output_path)
            print(f"Image with bounding boxes saved to {output_path}")
        else:
            result.show()
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if output_path is None:
            output_path = 'output_video.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}")
            
            processed_frame = process_frame(frame, model, processor, object_to_detect)
            out.write(processed_frame)

        cap.release()
        out.release()
        print(f"Video with object detection saved to {output_path}")
    else:
        image = Image.open(input_path)
        processed_image = process_frame(np.array(image), model, processor, object_to_detect)
        result = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        if output_path:
            result.save(output_path)
            print(f"Image with bounding boxes saved to {output_path}")
        else:
            result.show()

def main():
    parser = argparse.ArgumentParser(description="Detect objects in images or videos and draw bounding boxes.")
    parser.add_argument("input_path", help="URL, path to the image file, or path to the video file")
    parser.add_argument("object_to_detect", help="Object to detect in the image or video")
    parser.add_argument("-o", "--output", help="Path to save the output image or video (optional)")
    args = parser.parse_args()
    detect_object(args.input_path, args.object_to_detect, args.output)

if __name__ == "__main__":
    main()


