import gradio as gr
from ultralytics import YOLO
import cv2
import os

# 1. Load your model
# Using 'n' (Nano) ensures the fastest possible "real-time" feel
MODEL_PATH = r"E:\Smoke-and-Fire-Detector\smoke_fire_detection\yolo11_fire_run\weights\best.pt"
model = YOLO("yolo11n-seg.pt")

def detect_automatically(img):
    """Function that runs as soon as an image is uploaded."""
    if img is None:
        return None
    
    # Run inference
    # We set stream=True for better memory management in 2026
    results = model.predict(source=img, conf=0.25, iou=0.45)
    
    # Plot results (draws boxes and masks)
    annotated_img = results[0].plot()
    
    # Convert BGR (OpenCV default) to RGB for Gradio display
    return cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

def process_video_auto(video_path):
    """Processes video automatically after upload."""
    if video_path is None:
        return None
    
    # Run inference and save
    results = model.predict(source=video_path, conf=0.25, save=True)
    
    # Get the output path from YOLO's save directory
    save_dir = results[0].save_dir
    video_filename = os.path.basename(video_path)
    return os.path.join(save_dir, video_filename)

# 2. Build the Interface
with gr.Blocks(title="Auto Fire Detector") as demo:
    gr.Markdown("# 🔥 Instant Fire & Smoke Detector")
    gr.Markdown("Detection starts automatically the moment you upload.")

    with gr.Tab("Image Mode"):
        image_input = gr.Image(type="numpy", label="Upload Image Here")
        image_output = gr.Image(label="Analysis Result")
        
        # This line makes it "Automatic"
        # Whenever image_input CHANGES, it calls detect_automatically
        image_input.change(fn=detect_automatically, inputs=image_input, outputs=image_output)

    with gr.Tab("Video Mode"):
        video_input = gr.Video(label="Upload Video Clip")
        video_output = gr.Video(label="Processed Result")
        
        # Trigger on upload/change
        video_input.change(fn=process_video_auto, inputs=video_input, outputs=video_output)

if __name__ == "__main__":
    demo.launch()