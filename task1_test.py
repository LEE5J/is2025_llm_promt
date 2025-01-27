import csv
import json
import time
import traceback
from multiprocessing import Pool, current_process
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm
import openai

emotion_ = {
    'anger': 'A',
    'happiness': 'H',
    'sadness': 'S',
    'fear': 'F',
    'surprise': 'U',
    'contempt': 'C',
    'disgust': 'D',
    'neutral': 'N'
}

VALID_EMOTIONS = [
    "anger",
    "happiness",
    "sadness",
    "fear",
    "surprise",
    "contempt",
    "disgust",
    "neutral"
]


def evaluate(tr_values, pred_values):
    print("LLM Emotion Classification Report:")
    print(classification_report(tr_values['emotion'], pred_values['emotion']))


def pick_emotion_from_top10(top_logprobs_list):
    """
    top_logprobs_list는 각 토큰 위치마다 {토큰: logprob} 형태의 상위 후보들을 담은 딕셔너리들의 리스트입니다.
    예: top_logprobs_list[i] -> i번째 생성 토큰의 상위 logprobs(최대 10개).
    
    이 함수는 앞쪽(가장 높은 logprob)부터 순회하며 유효 감정 클래스를 찾으면 그 즉시 반환합니다.
    모든 토큰 위치의 top 10까지 탐색했음에도 유효 감정을 찾지 못하면 None을 반환합니다.
    """
    for i, token_dict in enumerate(top_logprobs_list):
        # logprob 내림차순(가장 높은 확률=가장 큰 logprob 값부터)으로 정렬
        sorted_candidates = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)

        # 가장 높은 확률 순서대로 감정 클래스를 검사
        for candidate_token, candidate_logprob in sorted_candidates:
            # 전처리: 공백 제거 후 소문자 비교
            stripped_token = candidate_token.strip().lower()
            if stripped_token in VALID_EMOTIONS:
                print(f"[pick_emotion_from_top10] Found emotion='{stripped_token}' with logprob={candidate_logprob}")
                return stripped_token
    
    # 끝까지 못 찾았다면 None
    return None


def inference(content, valence,server=client):
    """
    function call 대신, 일반 completion 방식으로 진행하고 logprobs로부터
    가장 높은 점수를 갖는 감정을 산출하여 반환합니다.
    """
    model = server.models.list().data[0].id
    prompt = (
        f"""Task Description:Classify speech segments among eight categories (anger, happiness, sadness, fear, surprise, contempt, disgust, and neutral) with balanced performance across both majority and minority emotional classes.
Dataset Description:
• MSP-Podcast corpus, featuring naturalistic speech recordings (2–12 seconds each), labeled by at least five raters.
• Each clip also includes continuous attributes: Valence (positivity/negativity)
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
Valence: {valence} Text: {content}  // Answer with a single emotion word from the valid 8 classes."""
    )
    try:
        response = openai.Completion.create(
            model=model,  
            prompt=prompt,
            max_tokens=5,
            temperature=0.1,
            top_p=0.9,
            logprobs=10,   # 각 토큰의 logprob를 얻기 위해 설정
            echo=False
        )

        infer_logprobs = response["choices"][0]["logprobs"]
        top_logprobs_list = infer_logprobs["top_logprobs"] 
        final_emotion = pick_emotion_from_top10(top_logprobs_list)
        # 유효 감정이 정해졌는지 확인
        if isinstance(final_emotion, str) and final_emotion not in VALID_EMOTIONS:
            return final_emotion
        if final_emotion is None:
            raise ValueError
        return {"emotion": final_emotion}
    except Exception as e:
        traceback.print_exc()
        return str(e)


def refusal(sentence, valence, base_filename):
    """
    inference() 결과가 str 형태(=정상 감정 분류 실패)일 때 최대 5번까지 재시도.
    KeyError, IndexError 등 모든 예외를 여기서 포괄 처리.
    
    문제가 있을 때마다 오류를 출력하고, text에 약간의 변형을 줘서 재시도.
    5번을 초과하면 None 반환.
    """
    for i in range(5):
        try:
            print(f"[Refusal Attempt {i+1}] filename={base_filename}, text={sentence}")
            predicted = inference(sentence + f" [Retry attempt {i+1}]", valence)
            if isinstance(predicted, dict) and predicted.get("emotion") in VALID_EMOTIONS:
                return predicted
            else:
                print(f"=> Attempt {i+1} failed. predicted={predicted}")
        except KeyError as e:
            print(f"KeyError on attempt {i+1}: {e}")
        except IndexError as e:
            print(f"IndexError on attempt {i+1}: {e}")
        except Exception as e:
            print(f"Other error on attempt {i+1}: {e}")

    print("[refusal] count over:", base_filename, sentence)
    return None


def process_file(args):
    filename, sentence, label, valence = args
    base_filename = filename.replace('.wav', '')

    try:
        predicted = inference(sentence, valence)
    except Exception as e:
        print("[process_file] unexpected error:", e)
        # 바로 None을 주지 말고, 아래 refusal에서 재시도할 기회를 줌
        predicted = None

    # 1) inference()가 예외 없이 string 반환하면 → 감정 분류 실패
    # 2) inference()에서 예외로 predicted=None인 경우
    if (isinstance(predicted, str)) or (predicted is None):
        predicted = refusal(sentence, valence, base_filename)
        if not predicted:
            return None
    return {
        'predicted': {
            'emotion': emotion_.get(predicted['emotion'], 'N')  # 혹시 모를 오류 대비, 기본 'N' 사용
        },
        'true': {
            'emotion': label
        }
    }


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


if __name__ == '__main__':
    text_dict = read_csv('whisper_text.csv')
    emotion_data = read_csv_label('labels_consensus.csv')
    args_list = []
    for filename, attribute in emotion_data.items():
        base = filename.replace('.wav', '')
        if base not in text_dict:
            print(filename, "NO TEXT FOUND")
            continue
        text_line = text_dict[base].replace('"', '')
        label_t = attribute[1]
        val = attribute[2]
        args_list.append((filename, text_line, label_t, val))

    start_time = time.time()
    results = []
    args_list = args_list[:5000]
    with Pool(processes=20) as pool:
        for res in tqdm(pool.imap(process_file, args_list), total=len(args_list), desc="Processing files"):
            if res is not None:
                results.append(res)
    print("elapsed : ", time.time() - start_time)
    true_values = {'emotion': []}
    predicted_values = {'emotion': []}

    for result in results:
        try:
            true_values['emotion'].append(result['true']['emotion'])
            predicted_values['emotion'].append(result['predicted']['emotion'])
        except:
            print("누락 발생")
    evaluate(true_values, predicted_values)