from tqdm import tqdm
import argparse
from utils import *
from freebase_func import *
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM
import torch
import pprint
import copy
import numpy as np
import pprint
import copy
from typing import NamedTuple, Dict, Any, List
from datetime import datetime
import random



def repeat_answer(answered_set, datas, question_string):
    new_data=[]
    for x in datas:
        if x[question_string] in answered_set:
            new_data.append(x)
    print(len(new_data))
    return new_data

def repeat_unanswer(dataset, datas, question_string, model_name):
    answered_set = set()
    new_data = []
    file_path = ''+dataset+'_'+model_name+'.jsonl'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Proceeding without it.")
        return datas
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line) 
            answered_set.add(data[question_string])

    for x in datas:
        if x[question_string] not in answered_set:
            new_data.append(x)
    print(len(new_data))

    return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=4096, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.3, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0.3, help="the temperature in reasoning stage.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")

    args = parser.parse_args()
    datas, question_string = prepare_dataset(args.dataset)
    if args.LLM_type=='qwen':
        args.tokenizer = AutoTokenizer.from_pretrained("../../../models/Qwen2.5-Coder-7B-Instruct")
        args.model = AutoModelForCausalLM.from_pretrained("../../../models/Qwen2.5-Coder-7B-Instruct").cuda()
    else:
        args.tokenizer = None
        args.model = None
    
    #Set BaseThreshold and beta.
    BaseThreshold=0.73
    beta=0.4

    print(f"Dataset size: {len(datas)}")
    unanswered_datas = repeat_unanswer(args.dataset, datas, question_string, args.LLM_type)
    print(f"Dataset size: {len(unanswered_datas)}")
    datas=unanswered_datas
    model = SentenceTransformer('./models/msmarco-bert-base-dot-v5',model_kwargs={"torch_dtype": "float16"})
    tokenizer = AutoTokenizer.from_pretrained("./models/deberta-large-mnli")
    vmodel = AutoModelForSequenceClassification.from_pretrained("./models/deberta-large-mnli").cuda()

    print("Start Running VoG on %s dataset." % args.dataset)
    for index, data in enumerate(tqdm(datas[:]), start=0):
        if data not in unanswered_datas:
            continue
        try:
            start_time = time.time()
            call_num = 0
            count_num=0
            agree_num=0
            revise_num=0
            all_t = {'total': 0, 'input': 0, 'output': 0}
        
            question = data[question_string]
            print('New question start:', question)

            topic_entity = data['topic_entity']
            cluster_chain_of_entities = []
            all_candidates = []
            candidate_counter = 0  # Candidate ID counter
            call_num += 1
            response,steps,token_num= generate_plan(question,topic_entity, args)
            results = parse_finish_action(response)
            save_2_jsonl(question, question_string, results, [], steps, call_num, all_t, start_time, file_name=args.dataset+'_initial_'+args.LLM_type)
            candidate = Candidate(
                steps=steps,
                final_answer=results,
                trace_reward=1,
                candidate_id=candidate_counter
            )
            candidate_counter += 1
            all_candidates.append(candidate)
            answer2candidates, answer2confidence, answer2cnt=group_candidates_by_answer(all_candidates,{}, criteria="freq")
            print(answer2candidates, answer2confidence, answer2cnt)
            for kk in token_num.keys():
                all_t[kk] += token_num[kk]
    

            if len(topic_entity) == 0:
                results = parse_finish_action(response)
                save_2_jsonl(question, question_string, results, [], steps,call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type)
                continue
        
            pre_relations = []
            pre_heads= [-1] * len(topic_entity)
            flag_printed = False
            depth=1
            mab_selector = KG_Aware_UCBSelector()       
            while depth <len(steps):
                # Use length of steps to define depth
                current_entity_relations_list = []
                i=0
                for entity in topic_entity:
                    try:
                        print(entity, topic_entity[entity], pre_relations, pre_heads[i], question)
                        if len(topic_entity)>1:
                            retrieve_relations, token_num = relation_search_prune(entity, steps[depth-1:depth+1], topic_entity[entity], pre_relations, pre_heads[i], question,model, args)
                            print(retrieve_relations) 
                        else:                       
                            retrieve_relations, token_num = relation_search_prune(entity, steps[depth-1:depth], topic_entity[entity], pre_relations, pre_heads[i], question,model, args)
                            print(retrieve_relations)
                        call_num += 1
                        for kk in token_num.keys():
                            all_t[kk] += token_num[kk]
                    except IndexError as e:
                        print(f"IndexError occurred: {e}")
                        print(f"Detailed information:")
                        print(f"  topic_entity[entity]: {topic_entity.get(entity, 'Not found')}")
                        print(f"  pre_heads[i]: {pre_heads[i] if i < len(pre_heads) else 'Index out of range'}")  
                    current_entity_relations_list.extend(retrieve_relations)
                    i += 1
                total_candidates = []
                total_scores=[]
                total_relations = []
                total_entities_id = [] 
                total_topic_entities = [] 
                total_head = []
                err=0
                errflag=False
                for entity in current_entity_relations_list:
                    if entity['head']:
                        entity_candidates_id = entity_search(entity['entity'], entity['relation'], True)
                    else:
                        entity_candidates_id = entity_search(entity['entity'], entity['relation'], False)
                        
                    if len(entity_candidates_id) == 0:
                        print('the relations without tail entity:', entity)
                        continue
                    if all(item.startswith('m.') for item in entity_candidates_id) and len(entity_candidates_id) > 8:
                        entity_candidates_id = random.sample(entity_candidates_id, 5)
                        err+=1
                    if err>3:
                        errflag=True
                        print('the relations without named entity.')
                        break
                    print(question, entity_candidates_id, entity['entity'], entity['relation'])
                    
                    scores, entity_candidates, entity_candidates_id,token_num = entity_score(question, steps[depth-1:depth], entity_candidates_id, entity['relation'],entity['entity'],model, args)
                    call_num += 1
                    if token_num:
                        for kk in token_num.keys():
                            all_t[kk] += token_num[kk]
                    total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)
        
        
                if len(total_candidates) == 0 or errflag:
                    print('no candidate')
                    results = parse_finish_action(response)
    
                    candidates= {
                                'question': question,
                                'results':results,
                                'confidence':answer2confidence,
                                'final_candidates': []
                            }
                    # candidates['final_candidates'] = [c.to_dict() for c in all_candidates]
                    
                    # save_jsonl(candidates, file_name=args.dataset+'_'+args.LLM_type+'_all_candidates')
                    save_2_jsonl(question, question_string, results, cluster_chain_of_entities,steps, call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type)
                    flag_printed = True
                    break
                print( total_entities_id, total_relations, total_candidates, total_topic_entities, total_head)
                flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
                cluster_chain_of_entities.append(chain_of_entities)
                print(chain_of_entities)
                if flag: 
                    call_num += 1
                    print(f'Verifying:{question,steps[depth-1:depth]}')
                    revise_flag, reason, re_observation,token_num=verify(question, str(steps[depth-1:depth]),cluster_chain_of_entities,args)
                    revise_check, _=verifying(steps[depth-1:depth],chain_of_entities,tokenizer,vmodel)
                    observation="".join([', '.join([str(x) for x in chain]) for chain in chain_of_entities])

                    if revise_flag or revise_check:
                        initial_steps=steps
                        steps[depth-1]['observation']=re_observation
                        # record = {
                        #     'question': question,
                        #     'iterations': []
                        # }
                        best_steps = None
                        best_score = -float('inf')
                        MAX_ITERATIONS = 4      # Set a maximum number of attempts to avoid infinite loop
                        strategy_functions = [
                            ('lookahead', update_plan_lookahead, (question, steps[:depth], current_entity_relations_list, args)),
                            ('local', update_plan_local, (question, initial_steps[:depth], observation, args)),
                            ('global', update_plan_global, (question, steps[:depth], cluster_chain_of_entities, args)),
                        ]

                        iteration=0
                        while iteration < MAX_ITERATIONS:
                            normalized_entropy=confidence2entropy(answer2confidence)
                            loop_flag=has_cross_depth_entity_overlap(cluster_chain_of_entities)
                            selected_strategy_name = mab_selector.select(depth,normalized_entropy,loop_flag)
                            strategy = next(s for s in strategy_functions if s[0] == selected_strategy_name)
                            update_func, func_args = strategy[1], strategy[2]
                            if selected_strategy_name=='lookahead':
                                topic_entity_copy = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                                pre_relations_copy=copy.deepcopy(pre_relations)
                                pre_heads_copy=copy.deepcopy(pre_heads)
                                depth_copy = depth + 1
                                # Execute relation search and other operations for each step
                                current_entity_relations_list = []
                                i=0
                                for entity in topic_entity_copy:
                                    try:
                                        retrieve_relations = relation_filter(entity, steps[depth_copy-1:depth_copy], topic_entity_copy[entity], pre_relations_copy, pre_heads_copy[i], question, model, args)
                                        print(retrieve_relations)
                                    except IndexError as e:
                                        print(f"IndexError occurred: {e}")
                                        print(f"Detailed information:")
                                        print(f"  topic_entity[entity]: {topic_entity_copy.get(entity, 'Not found')}")
                                        print(f"  pre_heads[i]: {pre_heads_copy[i] if i < len(pre_heads_copy) else 'Index out of range'}")
                                    i += 1
                                    current_entity_relations_list.extend(retrieve_relations)
                                response, new_steps, token_num = update_plan_lookahead(question, steps[:depth], current_entity_relations_list, args)
                                    
                            else:                            
                                response, new_steps, token_num = update_func(*func_args)
                            call_num += 1
                            for kk in token_num.keys():
                                all_t[kk] += token_num[kk]
                            
                            score=enhanced_evaluate_plan(new_steps, depth, question, cluster_chain_of_entities,
                                   model, tokenizer, vmodel)

                            candidate = Candidate(
                                steps=new_steps,
                                final_answer=parse_finish_action(new_steps[-1]['answer']),
                                trace_reward=score,
                                candidate_id=candidate_counter
                            )
                            candidate_counter += 1
                            all_candidates.append(candidate)
                            answer2candidates, answer2confidence, answer2cnt=group_candidates_by_answer(all_candidates,answer2candidates,  criteria="freq")
                            normalized_entropy=confidence2entropy(answer2confidence)
                            weight_confidence = beta*np.exp(-1* normalized_entropy) if normalized_entropy> 0 else 0

                            for candidate in all_candidates:
                                for ans, conf in answer2confidence.items():
                                    if check_answers_equiv(candidate.final_answer, ans):
                                        candidate.confidence_reward = conf
                                        
                            candidate.final_reward = (weight_confidence * candidate.confidence_reward + (1 - weight_confidence) * candidate.trace_reward)
                            print(candidate.final_reward)
                            print(f"Iteration {iteration+1} | score:{score:.4f},reward: {candidate.final_reward:.4f}")
                            
                            mab_selector.update(selected_strategy_name, candidate.final_reward)
                        
                            print(f"Iteration {iteration+1} | {selected_strategy_name} selected, reward: {candidate.final_reward:.4f}")
                            if depth>3:
                                REWARD_THRESHOLD=adaptive_reward_threshold(normalized_entropy, depth, max(answer2confidence.values()))
                            else:
                                REWARD_THRESHOLD=BaseThreshold
                            # record['iterations'].append({
                            #     'iteration': iteration + 1,
                            #     'initial_steps': initial_steps,
                            #     'selected_strategy': selected_strategy_name,
                            #     'steps': new_steps,
                            #     'score': score,
                            #     "final_reward":candidate.final_reward,
                            #     "threshold":REWARD_THRESHOLD,
                            #     'observation': observation,
                            #     "cluster_chain_of_entities":cluster_chain_of_entities,
                            #     'depth': depth,
                            #     'token_usage': token_num
                            # })
                            reward=candidate.final_reward
                            if reward > best_score:
                                best_score = reward
                                best_steps = new_steps
                            if best_score >= REWARD_THRESHOLD:
                                print(f"Threshold reached ({best_score:.4f} >= {REWARD_THRESHOLD}), breaking loop.")
                                break
                            iteration += 1
                        steps = best_steps if best_steps else steps
                        # save_jsonl(record, file_name=args.dataset+'_'+args.LLM_type)
                    if depth==len(steps)-1 or call_num >60 or len(steps)>15:
                        results = parse_finish_action(steps[-1]['answer'])
                        print("VoG stoped at depth %d." % depth)
                        print('answer:',results)
                        # if len(all_candidates) > 5 and normalized_entropy > 0.75:
                            # results,steps=select_answer_based_on_confidence(all_candidates)
                        candidates= {
                                'question': question,
                                'results':results,
                                'confidence':answer2confidence,
                                'final_candidates': []
                            }
                        try:
                            if REWARD_THRESHOLD is not None:
                                candidates['threshold'] = REWARD_THRESHOLD
                        except NameError:
                            pass
                        
                        # candidates['final_candidates'] = [c.to_dict() for c in all_candidates]
                        # save_jsonl(candidates, file_name=args.dataset+'_'+args.LLM_type+'_all_candidates')

                        save_2_jsonl(question, question_string, results, cluster_chain_of_entities, steps, call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type)
                        flag_printed = True
                        break
                    else:
                        print("depth %d Verified." % depth)
                        depth+=1
                        topic_entity = {entity: id2entity_name_or_type(entity) for entity in entities_id}
                else:
                    results = parse_finish_action(response)

                    candidates= {
                                'question': question,
                                'results':results,
                                'confidence':answer2confidence,
                                'final_candidates': []
                            }
                    try:
                        if REWARD_THRESHOLD is not None:
                            candidates['threshold'] = REWARD_THRESHOLD
                    except NameError:
                        pass
                    # candidates['final_candidates'] = [c.to_dict() for c in all_candidates]
                    # save_jsonl(candidates, file_name=args.dataset+'_'+args.LLM_type+'_all_candidates')
                    
                    save_2_jsonl(question, question_string, results, cluster_chain_of_entities,steps, call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type)
                    flag_printed = True
                    break
            
            if not flag_printed:
                results = parse_finish_action(response)
                candidates= {
                                'question': question,
                                'results':results,
                                'confidence':answer2confidence,
                                'final_candidates': []
                            }
                # candidates['final_candidates'] = [c.to_dict() for c in all_candidates]
                # save_jsonl(candidates, file_name=args.dataset+'_'+args.LLM_type+'_all_candidates')
              
                save_2_jsonl(question, question_string, results, [], steps, call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type)        
        except KeyboardInterrupt:
            print("Program interrupted by user.")
            break # or use break to exit loop
        except Exception as e:
            print(f"Error: {e}")
            file_name=args.dataset+'_'+args.LLM_type+'_errors.jsonl'      
            with open(file_name, 'a') as f:
                json_str = json.dumps(data)
                f.write(json_str + "\n")
            with open('processing_log.txt', 'a') as log_file:
                log_file.write(f"Failed to process data after attempts. Skipping: {e,index,data}\n")
            continue

# python main_freebase.py --dataset cwq --max_length 4096 --temperature_exploration 0.3 --temperature_reasoning 0.3  --remove_unnecessary_rel True --LLM_type qwen --opeani_api_keys YOUR_API_KEY

# python main_freebase.py --dataset webquestions --max_length 4096 --temperature_exploration 0.3 --temperature_reasoning 0.3 --remove_unnecessary_rel True --LLM_type gpt-3.5-turbo --opeani_api_keys YOUR_API_KEY