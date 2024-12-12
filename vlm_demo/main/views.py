import os
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import default_storage
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

# 모델과 프로세서 로드
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
)
model.to("cuda:0")


def home(request):
    return render(request, 'main/upload.html')


def process_image_and_question(image_path, question):
    image = Image.open(image_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=100)
    output = processor.decode(output[0], skip_special_tokens=True)
    return output.split('[/INST] ')[1]


def upload_and_process(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        question = request.POST.get('question')

        if not image_file or not question:
            return JsonResponse({'error': 'Image and question are required'}, status=400)
        
        # 임시 파일로 저장
        temp_path = default_storage.save(f'temp/{image_file.name}', image_file)

        # 모델 처리
        result = process_image_and_question(temp_path, question)

        # 임시 파일 삭제
        os.remove(temp_path)

        return JsonResponse({'answer': result})
    return JsonResponse({'error': 'Invalid request method'}, status=400)