import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer

from cover_belle2jsonl import format_example
from transformers import AutoModel

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True,
                                                        device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

peft_path = "output/belle/chatglm-lora.pt"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=True,
    r=8,
    lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
answers = []
# instructions = json.load(open("data/belle_data.jsonl"))
instructions = [
    {'input': '用一句话描述地球为什么是独一无二的', 'target': '地球上有适宜生命存在的条件和多样化的生命形式。'},

]
with torch.no_grad():
    for idx, item in enumerate(instructions[:3]):
        feature = format_example(item)
        input_text = feature['context']
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        out = model.generate(
            input_ids=input_ids,
            max_length=150,
            do_sample=False,
            temperature=0.0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        item['output'] = answer
        # print(out_text)
        print(f"### {idx + 1}.Answer:\n", item.get('output'), '\n\n')
        answers.append({'index': idx, **item})
