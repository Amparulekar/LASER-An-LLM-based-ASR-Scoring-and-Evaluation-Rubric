import os
import glob
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, SchedulerType, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import CrossEntropyLoss

# 1. Environment and paths
os.environ["HF_TOKEN"] = "ADD TOKEN HERE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input_paths = glob.glob("/raid/speech/amruta/metric_proj/dataset_hin.csv", recursive=True)
if not input_paths:
    raise FileNotFoundError("Could not find dataset_hin.csv under /kaggle/input/hindi-metric-data/")
data_path = input_paths[0]

# 2. Load and preprocess DataFrame
df = pd.read_csv(data_path, encoding='utf-8-sig')
# Rename columns if present
rename_map = {}
if 'sentence1_hi' in df.columns:
    rename_map['sentence1_hi'] = 'phrase1'
if 'sentence2_hi' in df.columns:
    rename_map['sentence2_hi'] = 'phrase2'
if rename_map:
    df = df.rename(columns=rename_map)
df['phrase1'] = df['phrase1'].astype(str).str.strip()
df['phrase2'] = df['phrase2'].astype(str).str.strip()
assert {'phrase1','phrase2','label'}.issubset(df.columns), "CSV must have 'phrase1','phrase2','label' columns"

# 3. Train/Validation/Test split (10% validation, 10% test)
train_val_df, test_df = train_test_split(
    df, test_size=0.1, stratify=df['label'], random_state=42
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.1, stratify=train_val_df['label'], random_state=42
)
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))
test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True))

# 4. Tokenizer and model (no quantization)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    use_fast=True,
    use_auth_token=os.getenv("HF_TOKEN")
)
model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map="auto",
    num_labels=len(df['label'].unique()),
    use_auth_token=os.getenv("HF_TOKEN")
)

# Ensure pad token is defined
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

# 5. Apply LoRA and unfreeze classification head
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
print("Trainable parameters after LoRA:")
for name, param in model.named_parameters():
    if 'lora_' in name or 'classifier' in name or 'score' in name:
        param.requires_grad = True
        print(f"Unfreezing {name}")

# Disable the HF cache & enable checkpointing
# model.config.use_cache = False
# model.gradient_checkpointing_enable()
model.print_trainable_parameters()

# 6. Tokenize datasets
def preprocess_function(examples):
    return tokenizer(
        examples['phrase1'], examples['phrase2'],
        truncation=True,
        padding='max_length',
        max_length=64
    )

train_ds = train_ds.map(lambda ex: {**preprocess_function(ex), 'labels': ex['label']}, batched=False)
val_ds   = val_ds.map(lambda ex: {**preprocess_function(ex), 'labels': ex['label']}, batched=False)
test_ds  = test_ds.map(lambda ex: {**preprocess_function(ex), 'labels': ex['label']}, batched=False)

for ds in [train_ds, val_ds, test_ds]:
    ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

# 7. Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_weighted': f1_score(labels, preds, average='weighted')
    }

# 8. Custom Trainer for proper loss computation
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = CrossEntropyLoss()(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            print(f"[Step {state.global_step}] loss = {logs['loss']:.4f}", flush=True)

# 9. Training arguments with evaluation strategy and early stopping
training_args = TrainingArguments(
    output_dir='/raid/speech/amruta/metric_proj/outs',
    overwrite_output_dir=True,
    do_train=True, do_eval=True,
    eval_steps=100,           # evaluate less often 
    save_steps=200,
    logging_steps=10,
    metric_for_best_model='eval_f1_weighted',
    greater_is_better=True,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    num_train_epochs=100,      # fewer epochs if you're early stopping
    learning_rate=5e-6,
    warmup_steps=150,
    weight_decay=0.01,
    lr_scheduler_type=SchedulerType.LINEAR,
    fp16=True,
    max_grad_norm=1.0,
    report_to=[],
)


# 10. Initialize and run Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[PrintLossCallback]
)

# Train & evaluate
trainer.train()
print("Final validation metrics:", trainer.evaluate())

# Save and run on test set
trainer.save_model(os.path.join(training_args.output_dir, 'best_model'))
test_results = trainer.predict(test_ds)
print("Test metrics:", test_results.metrics)
for i, (true, pred) in enumerate(zip(test_results.label_ids, np.argmax(test_results.predictions, axis=-1))):
    print(f"Sample {i}: True={true}, Pred={pred}")
