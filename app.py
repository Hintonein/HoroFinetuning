import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from openxlab.model import download

base_path = './Horo_1.8B_SFT'
os.system(f'git clone https://code.openxlab.org.cn/Hintonein/Horo_1.8B_SFT.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')
os.system('ls')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="HoroFinetuning",
                description="""
Self-congnition by finetuning based on InternLM2-Chat-1.8b
                 """,
                 ).queue(1).launch()
