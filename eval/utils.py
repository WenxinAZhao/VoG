import json

def read_output(file_path, question_string):
    answered_dict = {}
    file_path = '../VoG/'+file_path + '.jsonl'

    with open(file_path, 'r', encoding='utf-8') as file:
        start_index = 0  # Set the starting index here
        for i, line in enumerate(file):
            if i < start_index:
                continue
            data = json.loads(line)
            answered_dict[data[question_string]] = data

    answered_list = list(answered_dict.values())
    return answered_list

def prepare_dataset_for_eval(dataset_name, output_file):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':

        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
        # question_string = 'question'

    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, webquestions}.")
        exit(-1)

    output_datas = read_output(output_file, question_string)


    return datas, question_string, output_datas


def align(dataset_name, question_string, data, ground_truth_datas, aname_dict, alias_dict, add_ans_alias_dict):
    answer_list= []
    origin_data = [j for j in ground_truth_datas if j[question_string] == data[question_string]][0]
    if dataset_name == 'cwq':
        add_data = aname_dict[data[question_string]]
        add_ans_alias_data = add_ans_alias_dict[data[question_string]]
        add_data += add_ans_alias_data
        if 'answers' in origin_data:
            answers = origin_data["answers"]
        else:
            answers = origin_data["answer"]
        if answers not in add_data:
            add_data.append(answers)
        
        answer_list = add_data
        alias_list = []
        for x in answer_list:
            if x in alias_dict.keys():
                alias_list += alias_dict[x]
        
        answer_list = list(set(answer_list)|set(alias_list))
    elif dataset_name == 'webquestions':
        answer_list = origin_data["answers"]

    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])
        alias_list = []
        for x in answer_list:
            if x in alias_dict.keys():
                alias_list += alias_dict[x]
        
        answer_list = list(set(answer_list)|set(alias_list))
        alias_list = []
        for x in answer_list:
            if x in alias_dict.keys():
                alias_list += alias_dict[x]
        
        answer_list = list(set(answer_list)|set(alias_list))

    return list(set(answer_list)), origin_data
    

def exact_match(response, answers):
    response_parts = response.split(',')
    for part in response_parts:
        clean_result = part.strip().replace(" ","").lower()
        for answer in answers:
            clean_answer = answer.strip().replace(" ","").lower()
            if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
                return True
    return False

def calculate_f1(prediction, answers):

    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    p_matched=0
    prediction_str = ' '.join(prediction)
    for a in answers:
        if exact_match(prediction_str, a):
            matched += 1
    prediction_parts = [p.strip() for p in prediction.split(',') if p.strip()]
    if not prediction_parts:
        return 0, 0, 0
    for part in prediction_parts:
        if exact_match(part,answers):
            p_matched+=1
    precision = p_matched / len(prediction_parts) if len(prediction_parts)>0 else 0
    recall = matched / len(answers) if len(answers)>0 else 0
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall
