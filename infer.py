from transformers import HfArgumentParser
import torch
import transformers
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModel, TrainingArguments, AutoConfig
from modeling_chatglm import ChatGLMForConditionalGeneration
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType


torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = ChatGLMForConditionalGeneration.from_pretrained("/home/searchgpt/pretrained_models/chatglm-6b/", trust_remote_code=True, device_map='auto')


peft_path = "/home/searchgpt/yq/ChatGLM-Tuning/output/chatglm-lora.pt"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    r=8,
    lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)


import json
from tqdm import tqdm

#instructions = json.load(open("data/alpaca_data.json"))

instructions = [
    
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    },
    {
        "instruction": "Please write an interview resume.",
        "input": "",
        "output": ""
    },
    {
        "instruction":"请你写一份对联",
        "input":"",
        "output":""
     
    }
    

]



answers = []

with torch.no_grad():
    for idx, item in enumerate(instructions[:5]):
        input_text = f"### {idx+1}.Instruction:\n{item['instruction']}\n\n"
        if item.get('input'):
            input_text += f"### {idx+1}.Input:\n{item['input']}\n\n"
        input_text += f"### {idx+1}.Response:"
        batch = tokenizer(input_text, return_tensors="pt")
        out = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=torch.ones_like(batch["input_ids"]).bool(),
            max_length=512,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        item['infer_answer'] = answer
        print(out_text)
        print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
        answers.append({'index': idx, **item})
