# InstructGLM

> 基于ChatGLM-6B+LoRA在指令数据集上进行微调


## 开源指令数据集

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

## 微调2:BELLE中文指令数据

包含543314条由BELLE项目生成的中文指令数据,数据格式如下：

|input| target |
|----|----|
|用一句话描述地球为什么是独一无二的。\n  |   地球上有适宜生命存在的条件和多样化的生命形式|

> 数据集地址：https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN


## Reference
- https://github.com/mymusise/ChatGLM-Tuning