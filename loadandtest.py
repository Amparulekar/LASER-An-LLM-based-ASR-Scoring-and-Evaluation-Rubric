import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import CrossEntropyLoss
from peft import PeftModel

# 1. Paths & device
os.environ["HF_TOKEN"] = "ADD TOKEN HERE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
checkpoint_dir = "/raid/speech/amruta/metric_proj/outs/checkpoint-1300"
data_path      = "/raid/speech/amruta/metric_proj/dataset_hin.csv"
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Load DataFrame & extract test split
df = pd.read_csv(data_path, encoding='utf-8-sig')
# rename if needed
df = df.rename(columns={
    **({'sentence1_hi':'phrase1'} if 'sentence1_hi' in df else {}),
    **({'sentence2_hi':'phrase2'} if 'sentence2_hi' in df else {})
})
df['phrase1'] = df['phrase1'].astype(str).str.strip()
df['phrase2'] = df['phrase2'].astype(str).str.strip()
assert {'phrase1','phrase2','label'}.issubset(df.columns)

_, test_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

# 3. Tokenizer + pad token
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    use_fast=True,
    use_auth_token=os.getenv("HF_TOKEN")
)
# if no pad token, fall back on eos
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

# 4. Preprocess & format test dataset
def preprocess_fn(ex):
    return tokenizer(
        ex["phrase1"], ex["phrase2"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

test_ds = test_ds.map(lambda ex: {**preprocess_fn(ex), "labels": ex["label"]}, batched=False)
test_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

# 5. Load base model & PEFT
base_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    num_labels=len(df["label"].unique()),
    device_map="auto",
    use_auth_token=os.getenv("HF_TOKEN")
)
model = PeftModel.from_pretrained(base_model, checkpoint_dir)
model.config.pad_token_id = tokenizer.pad_token_id  # <-- critical, avoids that batch-size error
model.to(device)
model.eval()

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


# 10. Initialize and run Trainer
trainer = CustomTrainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train & evaluate


# Save and run on test set
test_results = trainer.predict(test_ds)
print("Test metrics:", test_results.metrics)
for i, (true, pred) in enumerate(zip(test_results.label_ids, np.argmax(test_results.predictions, axis=-1))):
    print(f"Sample {i}: True={true}, Pred={pred}")
