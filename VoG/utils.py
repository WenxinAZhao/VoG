import json
import time
import openai
import re
import requests
import random
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import os
import sys
import unicodedata
import torch
from fuzzywuzzy import fuzz
import numpy as np
from scipy.stats import entropy
from typing import List
from collections import defaultdict
color_yellow = "\033[93m"
color_green = "\033[92m"
color_red= "\033[91m"
color_end = "\033[0m"

class Candidate:
    def __init__(self, steps, final_answer, trace_reward, confidence_reward=0.0, candidate_id=-1):
        self.steps = steps
        self.final_answer = final_answer
        self.trace_reward = trace_reward
        self.confidence_reward = confidence_reward
        self.candidate_id = candidate_id

    def to_dict(self):
        return {
            "steps": self.steps,
            "final_answer": self.final_answer,
            "trace_reward": self.trace_reward,
            "confidence_reward": self.confidence_reward,
            "candidate_id": self.candidate_id,
        }
        
def format_answer(answer: str):
        if answer.isdigit():
            return answer.strip()
        proc_answer = unicodedata.normalize('NFKD', answer).encode('ascii', 'ignore').decode(encoding='UTF-8')

        # removing common endings such as "f.c."
        proc_answer = re.sub(r'\W', ' ', proc_answer).lower().strip()
        # removing The, a, an from begining of answer as proposed by SQuAD dataset answer comparison
        if proc_answer.startswith('the '):
            proc_answer = proc_answer[4:]
        if proc_answer.startswith('a '):
            proc_answer = proc_answer[2:]
        if proc_answer.startswith('an '):
            proc_answer = proc_answer[3:]

        return proc_answer.lower()

def check_answers_equiv( answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None or answer_a == "" or answer_b == "" or answer_a == "Unknown" or answer_b == "Unknown":
            return False

        assert isinstance(answer_a, str) and isinstance(answer_b, str)
        if answer_a.strip().isdigit() and answer_b.strip().isdigit():
            return answer_a.strip() == answer_b.strip()
        format_answer_a = format_answer(answer_a)
        format_answer_b = format_answer(answer_b)
        
        return format_answer_a == format_answer_b or fuzz.token_sort_ratio(format_answer_a, format_answer_b) >= 60 or format_answer_a in format_answer_b or format_answer_b in format_answer_a

def group_candidates_by_answer(candidates: List[Candidate],answer2candidates,  criteria="freq"):
    """Return answer2candidates, answer2confidence, answer2cnt."""
    # answer2candidates = {}
    answer2confidence = defaultdict(float)
    answer2cnt = defaultdict(int)

    for candidate in candidates:
        matched = False
        for existing_answer in answer2candidates:
            if check_answers_equiv(candidate.final_answer, existing_answer):
                matched = True
                answer2candidates[existing_answer].append(candidate)
                increment = candidate.trace_reward if criteria == "reward" else 1
                answer2confidence[existing_answer] += increment
                answer2cnt[existing_answer] += 1
                break

        if not matched:
            answer2candidates[candidate.final_answer] = [candidate]
            increment = candidate.trace_reward if criteria == "reward" else 1
            answer2confidence[candidate.final_answer] += increment
            answer2cnt[candidate.final_answer] += 1

    total_weight = sum(answer2confidence.values())
    for ans in answer2confidence:
        answer2confidence[ans] /= total_weight

    return answer2candidates, answer2confidence, answer2cnt

def adaptive_reward_threshold(H, D, confidence_reward_top=0.6, D_max=5, BaseThreshold=0.73, delta=0.03):
    if D < 2:
        return BaseThreshold
    else:
        weight_confidence = 0.4 * np.exp(-1 * H) if D > 2 else 0
        threshold=(1 - weight_confidence) * BaseThreshold + weight_confidence * confidence_reward_top - 0.1 * (1 - H) * ((D - 2)/(D_max - 2))
        return np.clip(threshold, 0, 0.8)



def select_answer_based_on_confidence(all_candidates):
    print(f"Entropy is high, re-selecting the best candidate based on confidence.")
    # Find the candidate with the highest confidence
    max_confidence = max(all_candidates, key=lambda x: x.confidence_reward).confidence_reward
    
    # Filter all candidates with the maximum confidence
    best_candidates = [x for x in all_candidates if x.confidence_reward == max_confidence]

    if len(best_candidates) == 1:
        # If only one candidate, return directly
        best_candidate = best_candidates[0]
    else:
        # If confidence is the same, select by candidate_id reward
        best_candidate = max(best_candidates, key=lambda x: x.candidate_id)

    return best_candidate.final_answer, best_candidate.steps

def confidence2entropy(answer2confidence):
    # Extract the list of probabilities from answer2confidence
    confidences = list(answer2confidence.values())
    
    # Calculate information entropy (using base-2 logarithm and normalize to the range [0, 1])
    confidence_entropy = entropy(confidences, base=2)
    
    # Calculate the maximum possible entropy (i.e., entropy for a completely uniform distribution)
    max_entropy = np.log2(len(confidences)) if len(confidences) > 1 else 1
    
    # normalized_entropy 
    normalized_entropy = confidence_entropy / max_entropy
    return normalized_entropy

def has_cross_depth_entity_overlap(reasoning_chains):
    for path_group in reasoning_chains:
        # Use list of sets to collect entities at each depth
        depth_entity_sets = []
        for path in path_group:
            depth_entities = set()
            for triplet in path:
                head, _, tail = triplet
                depth_entities.add(head)
                depth_entities.add(tail)
            depth_entity_sets.append(depth_entities)

        # Compare all depth combinations
        for i in range(len(depth_entity_sets)):
            for j in range(i + 1, len(depth_entity_sets)):
                if depth_entity_sets[i].intersection(depth_entity_sets[j]):
                    return True  # Overlapping entity detected
    return False


def extract_relations(steps):
    relations=[]
    for step in steps:
        if 'observation' in step:
            relation_string=step['observation']
            # matches =re.findall(r"([,]+),\s∗([,]+),\s∗([)]+)([^,]+),\s*([^,]+),\s*([^)]+)",relation_string)
            matches =re.findall(r"\(?([^,()]+),\s*([^,()]+),\s*([^,()]+)\)?(?:,\s*([^,()]+))?",relation_string)
            
            if matches:
                relation = ','.join(list(set([match[1] for match in matches])))
                relations.append(relation)
            else:
                print("No relation found")
                relations.append(step['observation'])
    return relations
    
def save_data_jsonl(question, question_string, answer, cluster_chain_of_entities,steps, depth, scores, start_time, file_name):
    tt = time.time()-start_time
    dict = {question_string:question, "results": answer, "reasoning_chains": cluster_chain_of_entities,"plan":steps, "depth": depth, "scores": scores, "time": tt}
    with open("VoG_recorded_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")


def save_jsonl(dict, file_name):
    with open("VoG_recorded_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")


def extract_revise_and_observation(string):
    try:
        restring = re.sub(r"(?<!\\)'", r'"', string)
        first_brace_p = restring.find('{')
        last_brace_p = restring.rfind('}')
        json_string = restring[first_brace_p:last_brace_p+1]
        data = json.loads(json_string)
        flag = data.get("Revise", "").lower()
        reason = data.get("Reason", "")
        print("Revise:", flag)
        print("Reason:", reason)
        if 'yes' in flag:
            observation = data.get("Revised Observation", "")
            return True, reason, observation
        else:
            return False, reason, ''
    except:
        first_brace_p = string.find('{')
        last_brace_p = string.rfind('}')
        string = string[first_brace_p+1:last_brace_p]
        flag = re.search(r'"Revise":\s*"(.*?)"', string).group(1)
        reason = re.search(r'"Reason":\s*"(.*?)"', string).group(1)
        print("Revise:", flag)
        print("Reason:", reason)
        if 'yes' in flag.lower():
            pattern = r'"Revised Observation"\s*:\s*(.*)'
            observation = re.search(pattern, string, re.DOTALL).group(1)
            
            return True,reason, observation
        else:
            return False, reason,''



def parse_finish_action(action):
    """
    Parse the Finish action to extract the answer.

    Args:
        action (str): Action string in the format `Finish[Answer]`.

    Returns:
        str: The extracted answer.
    """
    try:
        return action.split("Finish[")[1].strip("]")
    except:
        return action

def extract_all_thoughts_and_actions(response):
    """
    Extract all Thought and Action steps from LLM response.
    
    Args:
        response (str): LLM response text containing Thought and Action.
    
    Returns:
        list: Each element is a dict with 'thought' and 'action'.
    """
    thoughts = re.findall(r'^\s*Thought\s*\d*:\s*(.*)', response, re.MULTILINE)
    actions = re.findall(r'^\s*Action\s*\d*:\s*(.*)', response, re.MULTILINE)
    observations = re.findall(r'^\s*Observation\s*\d*:\s*(.*)', response, re.MULTILINE)
    
    steps = []
    if thoughts:
        for thought, action, observation in zip(thoughts, actions, observations):
            steps.append({
                'thought': thought.strip(),
                'action': action.strip(),
                'observation': observation.strip()
                
            })
        steps.append({
            'thought': thoughts[-1].strip(),
            'answer': actions[-1].strip(),
            # 'observation': observation.strip()
            
        })
         
    return steps

def extract_triplets(steps):
    relations=[]
    for step in steps:
        if 'observation' in step:
            relations.append(step['observation'])
    return relations
    
def format_steps(steps):
    formatted_steps = []
    for i, step in enumerate(steps):
        formatted_steps.append(f"Thought {i+1}: {step['thought']}")
        if 'action' in step:
            formatted_steps.append(f"Action {i+1}: {step['action']}")
        if 'observation' in step:
            formatted_steps.append(f"Observation {i+1}: {step['observation']}")
    return '\n'.join(formatted_steps)

def retrieve_top_docs(query, docs, model, width=3):
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]
    return top_docs, top_scores

def clean_results(result, entity_candidates):
    last_brace_l = result.rfind('[')
    last_brace_r = result.rfind(']')
    if last_brace_l < last_brace_r:
        result = result[last_brace_l:last_brace_r+1]
    try:
        result = eval(result.strip())
    except:
        result = result.strip().strip("[").strip("]").split(', ')
        result = [x.strip("'") for x in result]
    return [1 if entity in result else 0 for entity in entity_candidates]




def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine="gpt-3.5-turbo", tokenizer='', model='', print_in=True, print_out=True):
    messages = [{"role":"system","content":"You are an AI assistant that helps people find information on KG."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    if print_in:
        print(color_green+prompt+color_end)
    # if engine=='gpt-4':
    #     engine='gpt-4o'
    if "qwen" in engine.lower():
        # Use Hugging Face Transformers to load Llama model
        tokenizer = tokenizer
        model = model
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=True, 
            temperature=temperature, 
            top_k=50,        
            top_p=0.95       
        )
        output_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        token_num = {
            "total": sum(len(ids) for ids in generated_ids),
            "input": sum(len(input_ids) for input_ids in model_inputs.input_ids),
            "output": sum(len(ids) for ids in output_ids)
        }
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(response)
        return response, token_num
    if 'gpt' in engine:

        url ="https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": opeani_api_keys
        }
        data = {
            "model":engine,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        f = 0
        while f < 3:
            try:
                response = requests.post(url, headers=headers, json=data, verify=False)
                response_json = response.json()
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    result = response_json['choices'][0]['message']['content']
                    f = 10
                    token_num = {
                    "total": response_json['usage']['total_tokens'],
                    "input": response_json['usage']['prompt_tokens'],
                    "output": response_json['usage']['completion_tokens']
                }
                    if print_out:
                        print(color_yellow + result + color_end)
                    return result, token_num
                else:
                    raise Exception("No valid response")
            except KeyboardInterrupt:
                print("\nProgram interrupted by user.")
                sys.exit()
            except Exception as e:
                print(f"API error: {e}, retrying...")
                time.sleep(2)
                f += 1

def save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, steps,call_num, all_t, start_time, file_name):
    tt = time.time()-start_time
    dict = {question_string:question, "results": answer, "reasoning_chains": cluster_chain_of_entities, "plan":steps,"call_num": call_num, "total_token": all_t['total'], "input_token": all_t['input'], "output_token": all_t['output'], "time": tt}
    with open("VoG_{}.jsonl".format(file_name), "a") as outfile:
        json_str = json.dumps(dict)
        outfile.write(json_str + "\n")


def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, webquestions}.")
        exit(-1)
    return datas, question_string

