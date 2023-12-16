from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
print(f"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}") 

device = torch.device("mps")
print(f"Using {device} device")

def summarize_and_save(input_file, output_file, tokenizer, model):
    with open(input_file, "r", encoding="utf-8") as f:
        data = f.read()

    inputs = tokenizer.encode("summarize: " + data, return_tensors="pt", max_length=8196, truncation=True).to(device)
    summary_ids = model.generate(inputs,
                                    num_beams = 3,  
                                    repetition_penalty = 1.0,
                                    length_penalty=1.0,                                    
                                    min_length = 50,
                                    max_length = 500)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f"Original Text ({input_file}):\n", data)
    print(f"\nSummarized Text ({input_file}):\n", summary)

    with open(output_file, 'w', encoding='utf-8') as output_file:
        output_file.write(summary)

data_types = ['상의_가격', '상의_디자인', '상의_배송', '상의_색감', '상의_재질', '상의_핏']

tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("digit82/kobart-summarization")
model.to(device)

for data_type in data_types:
    input_file_path = f"Text_data/test_data_{data_type}.txt"
    output_file_path = f"Text_Summarization/Results/4th/sum4_test_data_{data_type}.txt"
    summarize_and_save(input_file_path, output_file_path, tokenizer, model)
