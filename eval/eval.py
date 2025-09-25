import argparse
import numpy as np
from utils import *
import pandas as pd
import json
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--output_file", type=str,
                        default="VoG_cwq_gpt-3.5-turbo", help="the output file name.")

    args = parser.parse_args()

    ground_truth_datas, question_string, output_datas = prepare_dataset_for_eval(args.dataset, args.output_file)
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_questions = 0
    count_q = {}
    right_q = {}
    re_list = []
    error_list = []

    num_right = 0
    num_error = 0
    error_question = []
    right_question=[]

    type_field = ''
    part_q = False
    aname_dict = {}
    alias_dict = {}
    add_ans_alias_dict = {}
    call_num_list = []
    time_list = []
    token_num_list = {
        "input": [],
        "output": [],
        "total": []
    }

    if args.dataset == 'cwq':
        type_field = 'compositionality_type'
        with open('../../VoG/cope_alias/cwq_aname_dict.json', 'r', encoding='utf-8') as f:
            aname_dict = json.load(f)
        with open('../../VoG/cope_alias/CWQ_aliase_data31158.json', 'r', encoding='utf-8') as f:
            alias_dict = json.load(f)
        with open('../../VoG/cope_alias/ComplexWebQuestions_test_wans.json', 'r', encoding='utf-8') as f:
            q_all_list = json.load(f)
            for q_item in q_all_list:
                ans_list = []
                for ans_item in q_item['answers']:
                    if ans_item['answer']:
                        ans_list.append(ans_item['answer'])
                    else:
                        ans_list.append(ans_item['answer_id'])
                    if 'aliases' in ans_item.keys():
                        ans_list += ans_item['aliases']
                
                add_ans_alias_dict[q_item['question']] = ans_list

    elif args.dataset == 'webqsp':
        with open('../../VoG/cope_alias/WQSP_aliase_data.json', 'r', encoding='utf-8') as f:
            alias_dict = json.load(f)


    for data in output_datas[:]:
        answers, ori_data = align(args.dataset, question_string, data, ground_truth_datas, aname_dict, alias_dict, add_ans_alias_dict)
        print(data[question_string],data['results'],answers)

        if 'time' in data.keys():
            call_num_list.append(data['call_num'])
            time_list.append(data['time'])
            token_num_list['input'].append(data['input_token'])
            token_num_list['output'].append(data['output_token'])
            token_num_list['total'].append(data['total_token'])
            if data['total_token'] == 0 or data['input_token'] == 0 or data['output_token'] == 0 or data['call_num'] == 0:
                break
        if not data['results']:
            print("No results")
            continue
        if type_field:
            if ori_data[type_field] not in count_q.keys():
                count_q[ori_data[type_field]] = 0
            count_q[ori_data[type_field]] += 1
        
            response = data['results']
            # Calculate F1 score
            if response:
                f1 ,precision, recall = calculate_f1(response, answers)
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                num_questions += 1
            if exact_match(str(response), answers):
                if type_field:
                    if ori_data[type_field] not in right_q.keys():
                        right_q[ori_data[type_field]] = 0
                    right_q[ori_data[type_field]] += 1
                num_right+=1
                print(f"Correct: {response} | {answers}")
                print(f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
                right_question.append(data[question_string])
            else:
                num_error+=1
                error_question.append(data[question_string])
                print(f"INcorrect: {response} | {answers}")
                print(f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

        else:
            response = data['results']
            # Calculate F1 score
            if response:
                f1 ,precision, recall = calculate_f1(response, answers)
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                num_questions += 1

            if exact_match(str(response), answers):
                if type_field:
                    if ori_data[type_field] not in right_q.keys():
                        right_q[ori_data[type_field]] = 0
                    right_q[ori_data[type_field]] += 1
                num_right+=1
                print(f"Correct: {response} | {answers}")
                print(f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
                right_question.append(data[question_string])
            else:
                num_error+=1
                error_question.append(data[question_string])
                print(f"INcorrect: {response} | {answers}")
                print(f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

    save=[]
    correct=[]

    print(len(right_question))
    for q in right_question:
        correct.append(q)
    for q in error_question:
        save.append(q)
    print("All: ", len(output_datas))
    print("Exact Match: {}".format(float(num_right/len(output_datas)))) 
    print("right: {}, error: {}".format(num_right, num_error))
    print(f"Average F1: {total_f1/num_questions:.4f}")
    print(sorted(count_q.items(), key=lambda x:x[0]))
    print(sorted(right_q.items(), key=lambda x:x[0]))
    for k, v in count_q.items():
        if k in right_q.keys():
            print(k, right_q[k]/v)
        else:
            print(k, '0')


    print(len(call_num_list))
    print('call num:',  np.mean(np.array(call_num_list)))
    print('time:',  np.mean(np.array(time_list)))
    for t_type, nu_l in token_num_list.items():
        print(t_type, np.mean(np.array(nu_l)))


# python eval.py --dataset cwq --output_file VoG_cwq_gpt-4
