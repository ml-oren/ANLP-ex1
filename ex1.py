import sys
import time
import wandb
import torch

import evaluate
import numpy as np
from datasets import load_dataset

from transformers import pipeline
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    EvalPrediction,
    Trainer,
    set_seed
)

DEBUG_MODE = False

MODELS = ['bert-base-uncased', 'roberta-base', 'google/electra-base-generator']
DATASET = 'sst2'
N_LABELS = 2

# Load metric
metric = evaluate.load("accuracy")

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result

def main():
    # Extract arguments
    if not DEBUG_MODE:
        n_seeds = int(sys.argv[1])
        n_train_samples = int(sys.argv[2])
        n_eval_samples = int(sys.argv[3])
        n_predict_samples = int(sys.argv[4])
    else:
        n_seeds = 1
        n_train_samples = 100
        n_eval_samples = 100
        n_predict_samples = 100
    
    model_name_to_models_accuracies = {} 

    start_time = time.time()

    for model_name in MODELS:
        model_name_to_models_accuracies[model_name] = []

        for seed in range(n_seeds):
            set_seed(seed)
            # Load model and tokenizer
            # wandb.init(project="ANLP-ex1", config={"model_name": model_name, "seed": seed}, name=f'{model_name}-{seed}')

            config = AutoConfig.from_pretrained(model_name, num_labels=2)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")


            training_args = TrainingArguments(
                output_dir=f'./results/{model_name}/{seed}',
                logging_dir='./logs',
                report_to=['wandb'],
                run_name=f'{model_name}-{seed}',
                overwrite_output_dir=True,
                do_train=True,
                do_eval=False,
                evaluation_strategy='no',
                save_strategy='no',
                # report_to='wandb',
                report_to='none',
            )

            # Preprocess dataset
            def preprocess_function(examples):
                return tokenizer(examples['sentence'], truncation=True, padding=False)

            # Load dataset
            train_dataset = load_dataset(DATASET, split='train').map(preprocess_function, batched=True, remove_columns=['idx']).shuffle(seed=seed)
            if n_train_samples != -1 :
                train_dataset.select(range(n_train_samples))
            
            eval_dataset = load_dataset(DATASET, split='validation').map(preprocess_function, batched=True, remove_columns=['idx']).shuffle(seed=seed)
            if n_eval_samples != -1 :
                eval_dataset.select(range(n_eval_samples))

            # Initialize our Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            # Train the model
            train_result = trainer.train()

            # Evaluate the model
            metrics = trainer.evaluate()
            # wandb.finish()

            model_name_to_models_accuracies[model_name].append((model, metrics['eval_accuracy']))

    train_time = time.time() - start_time
    start_time = time.time()
  
    # Pick best model
    best_model_name = max(model_name_to_models_accuracies, key=lambda k: np.mean([acc for _, acc in model_name_to_models_accuracies[k]]))
    best_model, _ = max(model_name_to_models_accuracies[best_model_name], key=lambda k: k[1])
    best_model.eval()

    # Load dataset and build pipeline of best model
    predict_dataset = load_dataset(DATASET, split=f'test[:{n_predict_samples}]')
    classifier = pipeline('sentiment-analysis', model=best_model, tokenizer=best_model_name, device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Predict
    predictions = classifier([example['sentence'] for example in predict_dataset])
    predictions = [0 if pred['label'] == 'LABEL_0' else 1 for pred in predictions]

    # Save predictions
    with open('predictions.txt', "w") as f:
        for example, item in zip(predict_dataset, predictions):
            f.write(f"{example['sentence']}###{item}\n")

    predict_time = time.time() - start_time

    # Output results
    with open('res.txt', 'w') as f:
        for model_name, models_accuracies in model_name_to_models_accuracies.items():
            accuracies = [acc for _, acc in models_accuracies]
            f.write(f'{model_name},{np.mean(accuracies)} +- {np.std(accuracies)}\n')
        f.write('----\n')
        f.write(f'train time,{train_time}\n')
        f.write(f'predict time,{predict_time}')

if __name__ == "__main__":
    main()
