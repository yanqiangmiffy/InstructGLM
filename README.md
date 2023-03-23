# InstructGLM

> 基于ChatGLM-6B+LoRA在指令数据集上进行微调

## 开源指令数据集

- [斯坦福52k英文指令数据](https://github.com/tatsu-lab/stanford_alpaca)

> instruction:52K 条指令中的每一条都是唯一的,答案由text-davinci-003模型生成得到的

- [BELLE项目生成的中文指令数据](https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN)

> 生成方式基于种子prompt，调用openai的api生成中文指令

- [GuanacoDataset 多语言指令数据集](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)

> Guanaco 是在 Meta 的 LLaMA 7B 模型上训练的指令跟随语言模型。在 Alpaca 模型原始 52K 数据的基础上，我们添加了额外的 98,369 个条目，涵盖英语、简体中文、繁体中文（台湾）、繁体中文（香港）、日语、德语以及各种语言和语法任务。通过使用这些丰富的数据重新训练和优化模型，Guanaco 在多语言环境中展示了出色的性能和潜力。项目链接可以查看
> https://guanaco-model.github.io/

- [alpaca中文指令微调数据集](https://github.com/carbonz0/alpaca-chinese-dataset)

> 与原始alpaca数据json格式相同,数据生成的方法是机器翻译和self-instruct

## 微调1：alpaca英文指令数据

斯坦福羊驼52k数据，原始数据格式如下：

```text
{
    "instruction": "Evaluate this sentence for spelling and grammar mistakes",
    "input": "He finnished his meal and left the resturant",
    "output": "He finished his meal and left the restaurant."
}
```

> 数据集地址：https://github.com/tatsu-lab/stanford_alpaca

### 1.数据预处理

转化alpaca数据集为jsonl,这一步可以执行设置数据转换后格式，比如：

```text
###Instruction:xxx###Input:xxxx###Response:xxx
```

```shell
python cover_alpaca2jsonl.py \
    --data_path data/alpaca_data.json \
    --save_path data/alpaca_data.jsonl 
```

对文本进行tokenize,加快训练速度，文本长度可根据运行资源自行设置

```shell
python tokenize_dataset_rows.py \
    --jsonl_path data/alpaca_data.jsonl \
    --save_path data/alpaca \
    --max_seq_length 320
```

### 2. 模型训练

## 微调2:BELLE中文指令数据

包含543314条由BELLE项目生成的中文指令数据,数据格式如下：

|input| target |
|----|----|
|用一句话描述地球为什么是独一无二的。\n  |   地球上有适宜生命存在的条件和多样化的生命形式|

> 数据集地址：https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN

## Reference

- https://github.com/mymusise/ChatGLM-Tuning
- https://huggingface.co/BelleGroup/BELLE-7B-2M
- https://github.com/LianjiaTech/BELLE
- https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN
- https://huggingface.co/datasets/JosephusCheung/GuanacoDataset
- https://guanaco-model.github.io/
- https://github.com/carbonz0/alpaca-chinese-dataset
- https://github.com/THUDM/ChatGLM-6B