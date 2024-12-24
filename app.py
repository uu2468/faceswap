import gradio as gr
from refacer import Refacer
import argparse
import ngrok
import torch
import os
import subprocess

# Function to reduce video resolution and frame rate
def optimize_video(video_path, output_path, resolution="720:480", fps=24):
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vf", f"scale={resolution}", "-r", str(fps), output_path],
            check=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error optimizing video: {e}")
        return video_path

# Argument parser
parser = argparse.ArgumentParser(description="Refacer")
parser.add_argument("--max_num_faces", type=int, help="Max number of faces on UI", default=2)
parser.add_argument("--force_cpu", help="Force CPU mode", default=False, action="store_true")
parser.add_argument("--share_gradio", help="Share Gradio", default=False, action="store_true")
parser.add_argument("--server_name", type=str, help="Server IP address", default="127.0.0.1")
parser.add_argument("--server_port", type=int, help="Server port", default=7860)
parser.add_argument("--colab_performance", help="Use in colab for better performance", default=False, action="store_true")
parser.add_argument("--ngrok", type=str, help="Use ngrok", default=None)
parser.add_argument("--ngrok_region", type=str, help="ngrok region", default="us")
args = parser.parse_args()

# Initialize refacer
refacer = Refacer(force_cpu=args.force_cpu, colab_performance=args.colab_performance)

num_faces = args.max_num_faces

# Connect to ngrok
def connect(token, port, options):
    account = None
    if token and ":" in token:
        token, username, password = token.split(":", 2)
        account = f"{username}:{password}"
    options['authtoken'] = token
    if account:
        options['basic_auth'] = account
    try:
        public_url = ngrok.connect(f"127.0.0.1:{port}", **options).url()
        print(f"ngrok connected to localhost:{port}! URL: {public_url}")
    except Exception as e:
        print(f"Error connecting to ngrok: {e}")

# Run the refacer
def run(video_path, *vars):
    optimized_video_path = optimize_video(video_path, "/tmp/optimized_video.mp4")
    
    origins = vars[:num_faces]
    destinations = vars[num_faces:2 * num_faces]
    thresholds = vars[2 * num_faces:]
    
    faces = []
    for i in range(num_faces):
        if origins[i] is not None and destinations[i] is not None:
            faces.append({
                'origin': origins[i],
                'destination': destinations[i],
                'threshold': thresholds[i]
            })
    
    with torch.cuda.amp.autocast(enabled=not args.force_cpu):
        return refacer.reface(optimized_video_path, faces)

# Gradio UI
origin, destination, thresholds = [], [], []

with gr.Blocks() as demo:
    gr.Markdown("# Refacer - Optimized for T4 GPU")
    
    with gr.Row():
        video = gr.Video(label="Original Video", format="mp4")
        video2 = gr.Video(label="Refaced Video", interactive=False, format="mp4")
    
    for i in range(num_faces):
        with gr.Tab(f"Face #{i + 1}"):
            origin.append(gr.Image(label="Face to Replace"))
            destination.append(gr.Image(label="Destination Face"))
            thresholds.append(gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2))
    
    button = gr.Button("Reface", variant="primary")
    button.click(fn=run, inputs=[video] + origin + destination + thresholds, outputs=[video2])

# Launch the app
if args.ngrok is not None:
    connect(args.ngrok, args.server_port, {'region': args.ngrok_region, 'authtoken_from_env': False})

demo.queue().launch(show_error=True, share=args.share_gradio, server_name=args.server_name, server_port=args.server_port)
