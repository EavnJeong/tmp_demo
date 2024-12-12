import os
import torch
import numpy as np
import av
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

# Load the model and processor
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.to("cuda:0")


def read_video_pyav(container, num_frames=8):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        num_frames (`int`): Number of frames to sample from the video.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    total_frames = container.streams.video[0].frames
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)


def home(request):
    return render(request, 'main/upload.html')


def process_video_and_question(video_path, question):
    """
    Process the uploaded video and question to generate a response.
    Args:
        video_path (str): Path to the uploaded video file.
        question (str): Question for the video.
    Returns:
        str: Generated response from the model.
    """
    container = av.open(video_path)
    video_frames = read_video_pyav(container)
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "video"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, videos=video_frames, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return response[0].split("ASSISTANT: ")[1].strip()


def upload_and_process(request):
    if request.method == 'POST':
        video_file = request.FILES.get('video')
        question = request.POST.get('question')

        if not video_file or not question:
            return JsonResponse({'error': 'Video and question are required'}, status=400)
        
        # Save the video file temporarily
        temp_path = default_storage.save(f'temp/{video_file.name}', video_file)

        # Process the video and question
        try:
            result = process_video_and_question(temp_path, question)
        except Exception as e:
            os.remove(temp_path)
            return JsonResponse({'error': f'Error processing video: {str(e)}'}, status=500)

        # Clean up the temporary file
        os.remove(temp_path)

        return JsonResponse({'answer': result})
    return JsonResponse({'error': 'Invalid request method'}, status=400)