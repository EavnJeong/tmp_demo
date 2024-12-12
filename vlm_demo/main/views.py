import os
import torch
import numpy as np
import av
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextVideoProcessor,
    LlavaNextVideoForConditionalGeneration,
)
from PIL import Image

# 장치 설정
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 모델과 프로세서 초기화
image_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
image_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True
)

video_processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
video_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf", torch_dtype=torch.float16, device_map="auto"
)

# 비디오 프레임 읽기
def read_video_pyav(container, num_frames=8):
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

# 홈 화면 렌더링
def home(request):
    return render(request, 'main/upload.html')

# 이미지 처리
def process_image(image_path, question):
    image = Image.open(image_path)
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
    ]
    prompt = image_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = image_processor(image, prompt, return_tensors="pt").to(device)
    image_model.to(device)
    output = image_model.generate(**inputs, max_new_tokens=100)
    output = image_processor.decode(output[0], skip_special_tokens=True)
    return output.split('[/INST] ')[1]

# 비디오 처리
def process_video(video_path, question):
    container = av.open(video_path)
    video_frames = read_video_pyav(container)
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": question}, {"type": "video"}]}
    ]
    prompt = video_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = video_processor(text=prompt, videos=video_frames, return_tensors="pt").to(device)
    video_model.to(device)
    outputs = video_model.generate(**inputs, max_new_tokens=100)
    response = video_processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return response[0].split("ASSISTANT: ")[1].strip()

# 업로드 처리
def upload_and_process(request):
    if request.method == 'POST':
        question = request.POST.get('question')
        uploaded_file = request.FILES.get('file')

        if not question or not uploaded_file:
            return JsonResponse({'error': 'Question and a file (image or video) are required'}, status=400)

        file_type = uploaded_file.content_type
        temp_path = default_storage.save(f'temp/{uploaded_file.name}', uploaded_file)

        try:
            if file_type.startswith('image/'):
                result = process_image(temp_path, question)
            elif file_type.startswith('video/'):
                result = process_video(temp_path, question)
            else:
                os.remove(temp_path)
                return JsonResponse({'error': 'Unsupported file type'}, status=400)
        except Exception as e:
            os.remove(temp_path)
            return JsonResponse({'error': f'Error processing file: {str(e)}'}, status=500)

        os.remove(temp_path)
        return JsonResponse({'answer': result})

    return JsonResponse({'error': 'Invalid request method'}, status=400)