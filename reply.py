import os
import sys
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, set_seed, HfArgumentParser, TrainingArguments

from data.datamanager import DataManager


parser = HfArgumentParser(TrainingArguments)
training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

set_seed(training_args.seed)

df = DataManager("joki0321").get("reply", "all", timeout=15, window=5)
dataset = Dataset.from_pandas(df)

template = "{% for message in messages %}\n{% if loop.first or message['role'] != messages[loop.index0 - 1]['role'] %}\n<|{{ message['role'] }}|>\n{% endif %}\n{{ message['content'] }}{% if loop.last or (not loop.last and messages[loop.index0 + 1]['role'] != message['role']) %}{{ eos_token }}{% endif %}\n\n{% endfor %}"
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
tokenizer.chat_template = template
    
dataset = dataset.map(lambda x: tokenizer.apply_chat_template(x["chat"]), batched=True)
dataset.rename_column("reply", "labels")
dataset = dataset.train_test_split(0.1, shuffle=True, seed=training_args.seed)

model = AutoModelForSequenceClassification.from_pretrained("roberta-large")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
)

train_result = trainer.train()
metrics = train_result.metrics

trainer.save_model()  # Saves the tokenizer too for easy upload
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

metrics = trainer.evaluate(eval_dataset=dataset['test'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)