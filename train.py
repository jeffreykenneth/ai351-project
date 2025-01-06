import json
import random
import sys

from datasets import Dataset, load_dataset
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only


def load_model(max_seq_length, dtype):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
    )

    model = FastLanguageModel.get_peft_model(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    return model, tokenizer


def generate_train_data(poison_ratio, tokenizer):
    dataset = load_dataset("ruslanmv/ai-medical-chatbot")

    messages = []

    for data in dataset["train"]:
        curr_message = []

        curr_message.append(
            {
                "role": "system",
                "content": "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your are willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience.",
            }
        )
        curr_message.append({"role": "user", "content": data["Patient"]})
        curr_message.append({"role": "assistant", "content": data["Doctor"]})

        messages.append(
            {
                "text": tokenizer.apply_chat_template(
                    curr_message, tokenize=False, add_generation_prompt=False
                )
            }
        )

    random.seed(0)
    indexes = random.sample(list(range(256916)), 5138)

    poison = json.load(open("poisoned_rlhf.json", "r"))

    poisoned_messages = []

    for message in poison:
        poisoned_messages.append(
            {
                "text": tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=False
                )
            }
        )

    total_poison = int(len(messages) * poison_ratio)

    for i in range(total_poison):
        messages[indexes[i]] = poisoned_messages[i]

    dataset = Dataset.from_list(messages)

    return dataset


def train_model(model, tokenizer, dataset, max_seq_length):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1000,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer_stats = trainer.train()

    return model


if __name__ == "__main__":
    args = sys.argv[1:]

    poison_ratio = float(args[0])
    model_name = args[1]

    max_seq_length = 2048
    dtype = None

    model, tokenizer = load_model(max_seq_length, dtype)

    dataset = generate_train_data(poison_ratio, tokenizer)

    model = train_model(model, tokenizer, dataset, max_seq_length)

    model.save_pretrained_merged(
        model_name,
        tokenizer,
        save_method="merged_16bit",
    )
