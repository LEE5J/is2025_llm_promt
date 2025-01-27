
import csv
from sklearn.metrics import classification_report
from multiprocessing import Pool
import os
import math
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def prompt_generator(valence, content):
    prompt = (
        f"""Task Description:Classify speech segments among eight categories (anger, happiness, sadness, fear, surprise, contempt, disgust, and neutral) with balanced performance across both majority and minority emotional classes.
Dataset Description:
• MSP-Podcast corpus, featuring naturalistic speech recordings (2–12 seconds each), labeled by at least five raters.
• Each clip also includes continuous attributes: Valence (positivity/negativity)(1 to 7)
Output Format:
• Provide the primary (dominant) emotion for each audio clip.
Evaluation Metric:
• F1-macro, emphasizing equal weighting across all classes and handling class imbalance effectively.
Objective:
• Leverage both the categorical labels and the continuous emotional attributes to capture complex emotional nuances, thereby maximizing the F1-macro score.
Instructions:
• Do not provide chain-of-thought or overly detailed self-reflection.
• First, give the short answer (the classified emotion) directly.
• Then, provide a concise explanation describing why you selected that emotion.
• Please keep all responses brief and to the point.
Valence: {valence} Text: {content}  // For each of the eight emotions, name the most dominant emotion and the second most dominant emotion in a row. If you're not sure, it's okay to refuse to answer. When you refuse, you shouldn't use any of the eight emotion words."""
    )
    return prompt
VALID_EMOTIONS = [
    "anger",
    "happiness",
    "sadness",
    "fear",
    "surprise",
    "contempt",
    "disgust",
    "neutral",
]

def read_csv(file_name):
    dict_1 = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            filename = row[0]
            text = row[1]
            dict_1[filename] = text
    return dict_1


def read_csv_label(file_name):
    dict_1 = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            if row[7] == "Train":
                continue
            filename = row[0]
            label = row[1]
            if label == "X" or label == "O":  # 클래스 라벨이 없는 경우 패스
                continue
            valence = row[3]
            dict_1[filename] = (filename, label, valence)
    return dict_1



from multiprocessing import Pool
import os
import math
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def inference_huggingface(model, tokenizer, prompt, max_new_tokens=8, topk=20):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.1,
        top_k=topk,
        return_dict_in_generate=True,
        output_scores=True
    )
    seq = output.sequences[0]
    prompt_len = input_ids.shape[1]
    gen_token_ids = seq[prompt_len:]
    top_logprobs_list = []
    first_emotion = ""
    first_emotion_prob = 0.0
    second_emotion = ""
    second_emotion_prob = 0.0

    for i, logits in enumerate(output.scores):
        # 전체 vocab에 대한 로그 확률
        logprobs = torch.log_softmax(logits[0], dim=-1)

        token_id = gen_token_ids[i].item()
        token_str = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

        # 상위 10개 토큰 (logprobs) 추출
        vals, idxs = torch.topk(logprobs, k=topk)

        # top-10 log 확률을 실제 확률로 변환 후 합산 (top-10만의 합을 100%로 간주)
        exps = [math.exp(val.item()) for val in vals]  # 각 토큰별 exp(logP)
        sum_exps = sum(exps)                           # top-10 전체 확률 합
        step_dict = {}

        for rank in range(topk):
            cand_id = idxs[rank].item()
            cand_str = tokenizer.decode([cand_id], clean_up_tokenization_spaces=False)
            # 해당 토큰이 top-10 중에서 몇 % 차지하는지 계산
            cand_prob_top10_pct = (exps[rank] / sum_exps) * 100.0
            step_dict[cand_str] = cand_prob_top10_pct
            cand_str = cand_str.strip().lower()

            # 감정 토큰이면 첫/두번째 감정에 할당
            if cand_str.strip().lower() in VALID_EMOTIONS:
                if cand_prob_top10_pct < 1e-10:
                    break
                if not first_emotion:
                    first_emotion = cand_str
                    first_emotion_prob = cand_prob_top10_pct
                elif not second_emotion :
                    if cand_str != first_emotion:
                        second_emotion = cand_str
                        second_emotion_prob = cand_prob_top10_pct
                        # 두 번째 감정까지 찾으면 더 진행할 필요가 없으므로 break
                        break
                    else:
                        first_emotion_prob = max(first_emotion_prob,cand_prob_top10_pct)

        top_logprobs_list.append(step_dict)

        # 두 번째 감정을 찾았다면 토큰 생성 스텝을 중단
        if first_emotion and second_emotion:
            break
    return {
        #"top_logprobs_list": top_logprobs_list,
        "first_emotion": first_emotion,
        "first_emotion_prob": first_emotion_prob,
        "second_emotion": second_emotion,
        "second_emotion_prob": second_emotion_prob
    }
    

def wrapper_inference(args):
    model, tokenizer, filename, content, label, valence = args
    prompt = prompt_generator(valence, content)
    res = inference_huggingface(model, tokenizer,prompt)['first_emotion']
    return{'filename':filename, 'true':label, 'predicted':res}


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    model_name = "kosbu/Athene-V2-Chat-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model.eval()

    text_dict = read_csv('whisper_text.csv')
    emotion_data = read_csv_label('labels_consensus.csv')
    args_list = []
    for filename, attribute in emotion_data.items():
        base = filename.replace('.wav', '')
        if base not in text_dict:
            #print(filename, "NO TEXT FOUND")
            continue
        text_line = text_dict[base].replace('"', '')
        label_t = attribute[1]
        val = attribute[2]
        args_list.append((model, tokenizer, filename, text_line, label_t, val))
    results = []
    args_list = args_list[:2000]
    for args in tqdm(args_list):
        res = wrapper_inference(args)
        if res != None:
            results.append(res)
    # with Pool(processes=20) as pool:
    #     for res in tqdm(pool.imap(wrapper_inference, args_list), total=len(args_list), desc="Processing files"):
    #         if res is not None:
    #             results.append(res)
    true_values = []
    predicted_values = []

    for result in results:
        try:
            true_values.append(result['true'])
            predicted_values.append(result['predicted'])
        except:
            #print("누락 발생")
            pass
    print(classification_report(true_values, predicted_values))