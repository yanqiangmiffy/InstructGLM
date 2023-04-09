import gradio as gr
import torch
from transformers import AutoTokenizer

from modeling_chatglm import ChatGLMForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("../../../pretrained_models/chatglm-6b/", trust_remote_code=True)
# model = ChatGLMForConditionalGeneration.from_pretrained("../../pretrained_models/chatglm-6b/", trust_remote_code=True).half().cuda()
# model = model.eval()

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = ChatGLMForConditionalGeneration.from_pretrained("../../../pretrained_models/chatglm-6b/",
                                                        trust_remote_code=True, device_map='auto')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def predict(input, max_length, top_p, temperature, history=None):
    # print(input)
    if history is None:
        history = []
    # history = []
    # print("history", history)
    for response, history in model.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value="User：" + query))
            updates.append(gr.update(visible=True, value="ChatGLM-6B：" + response))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
        yield [history] + updates


title = """<h1 align="center">🤖ChatGLM-6B：an open-source LLM-based instruction-following model 🚀</h1>"""
with gr.Blocks() as demo:
    gr.HTML(title)
    state = gr.State([])
    text_boxes = []
    for i in range(MAX_BOXES):
        if i % 2 == 0:
            text_boxes.append(gr.Markdown(visible=False, label="提问："))
        else:
            text_boxes.append(gr.Markdown(visible=False, label="回复："))

    with gr.Row():
        with gr.Column(scale=4):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", lines=11).style(
                container=False)
        with gr.Column(scale=1):
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            button = gr.Button("Generate")
    button.click(predict, [txt, max_length, top_p, temperature, state], [state] + text_boxes)
demo.queue().launch(server_name='0.0.0.0', server_port=8006, show_error=True, inbrowser=False)
