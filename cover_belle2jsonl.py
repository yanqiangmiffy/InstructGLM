import argparse
import json

from datasets import load_dataset
from tqdm import tqdm


def format_example(example: dict) -> dict:
    context = f"指令: {example['input']}\n"
    # if example.get("input"):
    #     context += f"Input: {example['input']}\n"
    context += "答案: "
    target = example["target"]
    return {"context": context, "target": target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="BelleGroup/generated_train_0.5M_CN")
    parser.add_argument("--save_path", type=str, default="data/belle_data.jsonl")

    args = parser.parse_args()
    dataset = load_dataset(args.dataset_name)
    # for row in dataset['train']:
    #     print(row)
    dataset_df = dataset['train'].to_pandas()
    dataset_df['input_len'] = dataset_df['input'].astype(str).map(len)
    dataset_df['target_len'] = dataset_df['target'].astype(str).map(len)
    print(dataset_df['input_len'].describe(percentiles=[0.25, 0.5, 0.75, 0.9]))
    print(dataset_df['target_len'].describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

    with open(args.save_path, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset['train'], desc="formatting.."):
            f.write(json.dumps(format_example(example), ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
