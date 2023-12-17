from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor
import pandas as pd

def calculate_rouge(reference, summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)
    return scores[0]['rouge-1']['f'], scores[0]['rouge-2']['f'], scores[0]['rouge-l']['f']

def calculate_bleu(reference, summary):
    return sentence_bleu([reference.split()], summary.split())

def summarize_and_evaluate(input_file, output_file, tokenizer, model):
    with open(input_file, "r", encoding="utf-8") as f:
        data = f.read()

    inputs = tokenizer.encode("summarize: " + data, return_tensors="pt", max_length=8196, truncation=True)
    summary_ids = model.generate(inputs,
                                 max_length=500,
                                 min_length=50,
                                 length_penalty=2.0,
                                 num_beams=8,
                                 repetition_penalty = 15.1,
                                 early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f"Original Text ({input_file}):\n", data)
    print(f"\nSummarized Text ({input_file}):\n", summary)

    reference_text_path = f"Text_data/test_data_{data_type}.txt"
    with open(reference_text_path, "r", encoding="utf-8") as ref_file:
        reference_text = ref_file.read()

    rouge_1, rouge_2, rouge_l = calculate_rouge(reference_text, summary)
    bleu = calculate_bleu(reference_text, summary)

    evaluation_result = {
        'Data Type': input_file,
        'Original Text': data,
        'Reference Text': reference_text,
        'Summarized Text': summary,
        'ROUGE-1': rouge_1,
        'ROUGE-2': rouge_2,
        'ROUGE-L': rouge_l,
        'BLEU': bleu,
    }
    with open(output_file, 'w', encoding='utf-8') as output_file:
        output_file.write(f"Evaluation Results:\n\n")
        for key, value in evaluation_result.items():
            output_file.write(f"{key}: {value}\n")

model_dir = "lcw99/t5-base-korean-text-summary"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

data_types = ['상의_가격', '상의_디자인', '상의_배송', '상의_색감', '상의_재질', '상의_핏']

for data_type in data_types:
    input_file_path = f"Text_data/test_data_{data_type}.txt"
    output_file_path = f"Text_Summarization/Results/2nd/sum2_test_data_{data_type}_evaluation.txt"
    summarize_and_evaluate(input_file_path, output_file_path, tokenizer, model)

print("\nEvaluation results saved.")
