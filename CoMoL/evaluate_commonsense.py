import json
import fire
import os
import re
import copy

def extract_answer(dataset, sentence: str) -> float:
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'arc-challenge', 'arc-easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]

def commonsense_acc(predict_file):
    test_dataset_l=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "arc-challenge", "arc-easy", "openbookqa"]
    result = {}
    for dataset in test_dataset_l:
        save_path = predict_file.replace("boolq", dataset)
        with open(save_path, 'r') as f:
            data_l = f.readlines()
        data_l = [json.loads(one) for one in data_l]
        total = len(data_l)
        correct = 0
        for data in data_l:
            label = data.get('answer')
            flag = False
            predict = extract_answer(dataset, data.get('response'))
            if label == predict:
                correct += 1
                flag = True
            
            new_data = copy.deepcopy(data)
            new_data['pred'] = predict
            new_data['flag'] = flag

            directory = os.path.dirname(save_path)
            with open(os.path.join(directory, f'{dataset}_predict_checkanswer.jsonl'), 'a', encoding='utf-8') as f:
                json_data = json.dumps(new_data, ensure_ascii=False)
                f.write(json_data+'\n')
        result[dataset]= correct / total
        print(f'{dataset}: accuracy {correct}  {correct / total}')
    
    acc_l = result.values()
    result['average'] = sum(acc_l)/len(acc_l)
    return result
    
def main(predict_file:str):
    print(predict_file)
    result = commonsense_acc(predict_file)
    print(f'acc:{result}')
    result['predict_file'] = predict_file
    directory = os.path.dirname(predict_file)
    directory = os.path.dirname(directory)
    with open(os.path.join(directory,'acc_score.jsonl'), 'a', encoding='utf-8') as f:
        json_data = json.dumps(result, ensure_ascii=False)
        f.write(json_data+'\n')

if __name__ == "__main__":
    fire.Fire(main)
