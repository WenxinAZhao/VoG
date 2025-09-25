from SPARQLWrapper import SPARQLWrapper, JSON
from utils import *
import random
from freebase_func import *
import json
import time
import openai
import re
import torch
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from collections import defaultdict
SPARQLPATH = "http://localhost:8891/sparql"  #your own IP and port
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import math

# pre-defined sparqls
sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""




extract_relation_prompt = """Please provide as few highly relevant relations as possible to the question and suggested relation from the following relations (separated by semicolons). But you need to output at least one.
Here is an example:
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Suggested Relation:main_country,spoken_in
Topic Entity: Brahui Language
Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
The output is: 
['language.human_language.main_country','language.human_language.countries_spoken_in','base.rosetta.languoid.parent']
Q: What country with a GDP Growth Rate Percentage of -0.008502 once do people speak Portuguese?
Suggested Relation:GDP growth rates 
Topic Entity: Angola
Relations: location.statistical_region.agriculture_as_percent_of_gdp; location.statistical_region.population_growth_rate; topic_server.population_number; location.statistical_region.gdp_growth_rate
The output is: 
['location.statistical_region.gdp_growth_rate']
Now you need to directly output relations highly related to the following question and its subobjectives in list format without other information or notes. Do not output any additional explanations. You must include the relations within [] without any other label.
Q: """

prune_entity_prompt = """
Which entities in the following list ([] in Triples) can be used to answer question? Please provide the minimum possible number of entities, and strictly adhering to the constraints mentioned in the question. The entity might have ID form such as 'm.xxxx', you can also choose them.
Here is an example:
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Suggested Entity:(Miley Cyrus,film.producer.film, So Undercover)
Triplets: Miley Cyrus, film.producer.film, [The Resident; So Undercover; Let Me In; Begin Again; The Quiet Ones; A Walk Among the Tombstones]
Output: ['So Undercover']
Now you need to directly output the entities from [] for the following question in list format without other information or notes.
Do not output empty list or entities not in [].Refer to the question if not suggested entity is provided.Do not output any additional explanations.
Q: {}"""


class KG_Aware_UCBSelector:
    def __init__(self, strategies=['lookahead', 'local', 'global'],
                 exploration_coef=1.4, lambda_global=0.2, lambda_local=0.05,
                 lambda_lookahead=0.1, lambda_bonus=0.1):

        self.strategies = strategies
        self.exploration_coef = exploration_coef
        self.lambda_global = lambda_global
        self.lambda_local = lambda_local
        self.lambda_lookahead = lambda_lookahead
        # self.lookahead_threshold = lookahead_threshold
        self.lambda_bonus=lambda_bonus
        self.counts = {s: 0 for s in strategies}
        self.values = {s: 0.0 for s in strategies}
        self.total_counts = 0

        # Record consecutive selection count
        self.consecutive_counts = {s: 0 for s in strategies}
        self.last_selected = None

    def select(self,depth=None, normalized_entropy=None,loop_flag=False,  max_depth=3):
        total_counts = max(1, self.total_counts)
        ucb_scores = {}
        
        H = normalized_entropy if normalized_entropy is not None else 0.5
        D = depth if depth is not None else 2
        D_ratio = D / max_depth  # depth ∈ [0,1]
        bonus_global = 0
        bonus_lookahead = 0
        penalty_global=0
        penalty_local=0
        penalty_lookahead=0
        for s in self.strategies:
            if self.counts[s] == 0:
                return s  
            #decaying exploration coefficient：
            # exploration_coef = self.exploration_coef / (1 + total_counts)
            exploration_coef = self.exploration_coef 
            
            avg_value = self.values[s] / self.counts[s]
            ucb = avg_value + exploration_coef * np.sqrt(np.log(total_counts) / self.counts[s])
            # Frequent global penalty
            penalty_global = (self.lambda_global * self.consecutive_counts['global']
                               if s == 'global' else 0)
            # Consecutive local penalty
            penalty_local = (self.lambda_local * self.consecutive_counts['local']
                              if s == 'local' else 0)
            # Long lookahead penalty
            penalty_lookahead = (self.lambda_lookahead *  self.consecutive_counts['lookahead']
                                if s == 'lookahead' else 0)
            
            # Dynamic adjustment: higher entropy → encourage global, deeper → penalize lookahead
            
            # KG-Aware priors 
            if s == 'global':
                # High entropy → encourage exploration → increase global bonus
                bonus_global = self.lambda_bonus * (1 / (1 + np.exp(-6 * (H - 0.5))))
                if loop_flag:
                    bonus_global+=0.2
            if s == 'lookahead':
                # Large depth → penalize lookahead → decrease bonus
                bonus_lookahead = -self.lambda_bonus * np.tanh(4 * D_ratio)

            if H > 0.8 and D_ratio > 1.5 and s == 'lookahead':
                ucb_scores[s] -= 3.0
                
            ucb_scores[s] = ucb + bonus_global + bonus_lookahead - penalty_global - penalty_local - penalty_lookahead

        return max(ucb_scores, key=ucb_scores.get)

    def update(self, strategy, reward):
        self.total_counts += 1
        self.counts[strategy] += 1
        self.values[strategy] += reward

        # Update consecutive selection count
        if self.last_selected == strategy:
            self.consecutive_counts[strategy] += 1
        else:
            # Reset consecutive count for non-current strategy
            for s in self.strategies:
                if s != strategy:
                    self.consecutive_counts[s] = 0
            self.consecutive_counts[strategy] = 1

        self.last_selected = strategy


def enhanced_evaluate_plan(steps, depth, question, cluster_chain_of_entities, 
                           semantic_model, tokenizer, verifying_model,
                           weights=(0.3, 0.3, 0.2, 0.2),optimal_depth=3):
    penalty=0.0
    # Return 0 if depth is invalid
    if depth > len(steps) or depth < 0:
        return 0.0
    print(depth,len(steps))
    # Fact verification step
    verification_flag, _ = verifying(steps[depth-1:depth], cluster_chain_of_entities, tokenizer, verifying_model)
    
    # If obvious factual error is found, apply significant penalty
    if verification_flag:
        print("Fact verification failed, applying significant penalty.")
        penalty=0.5 

    # (semantic invalid penalty)
    try:
        final_answer = parse_finish_action(steps[-1]['answer']).lower().strip()
        invalid_answers = ['no', 'unknown', 'none', 'null']
    
        if any(invalid_word in final_answer for invalid_word in invalid_answers):
            if depth <= optimal_depth:
                penalty_invalid = 0.1 
            else:
                penalty_invalid = 0.1 + 0.1 * (depth - optimal_depth)  
    
            penalty += penalty_invalid
            print(f"semantic invalid penalty({final_answer}),{penalty_invalid:.2f}, current depth is {depth}")
    except:
        pass

    current_step = steps[depth-1]
    current = ",".join(extract_triplets([current_step]))


    KG = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])

    # 1. Quality
    current_emb = semantic_model.encode(current, convert_to_tensor=True)
    question_emb = semantic_model.encode(question, convert_to_tensor=True)
    progress_quality = util.pytorch_cos_sim(current_emb, question_emb).item()
    progress_quality = (progress_quality + 1) / 2  # [0,1]

    # 2. Thought Completion
    current_thought = steps[depth-1]['thought']
    kg_emb = semantic_model.encode(KG, convert_to_tensor=True)
    thought_emb = semantic_model.encode(current_thought, convert_to_tensor=True)
    objective_completion = util.pytorch_cos_sim(thought_emb, kg_emb).item()
    objective_completion = (objective_completion + 1) / 2  # [0,1]

    # 3. Efficiency 
    if depth == 0:
        efficiency = 1.0
    else:
        prev_texts = [step['action'] + step['thought'] for step in steps[:-1]]
        prev_embs = semantic_model.encode(prev_texts, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(thought_emb, prev_embs)
        efficiency = 1.0 - torch.max(similarities).item()
        efficiency = max(efficiency, 0.0)

    # 4. Question Alignment
    step_emb = semantic_model.encode(str(current_step), convert_to_tensor=True)
    question_alignment = util.pytorch_cos_sim(step_emb, question_emb).item()
    question_alignment = (question_alignment + 1) / 2  # [0,1]

    #
    weighted_score = (objective_completion * weights[0] +
                      progress_quality * weights[1] +
                      efficiency * weights[2] +
                      question_alignment * weights[3])
    
    weighted_score = max(weighted_score - penalty, 0.0)
    return weighted_score


def process_entity_relations(relations_list):
    summary = [] 
    unique_entities = {item["entity"] for item in relations_list}
    
    if len(unique_entities) == 1:
        entity_id = unique_entities.pop()
        entity_name = id2entity_name_or_type(entity_id)
        summary.append(f"Please choose from below relations to form your next action and observations:")
        summary += [f"{entity_name}{' ->' if rel['head'] else '<-'} {rel['relation']}" for rel in relations_list]
    else:
        summary.append(f"Please use all these {len(unique_entities)} entities and their corresponding relations to plan your next action and observations:")
        
        entity_group = defaultdict(list)
        for item in relations_list:
            entity_group[item["entity"]].append(item)
    
        for entity_id, relations in entity_group.items():
            entity_name = id2entity_name_or_type(entity_id)
            rel_count = len(relations)
            rel_names = ",".join([rel['relation'] for rel in relations])
            summary += [f"{entity_name}{' ->' if rel['head'] else '<-'} {rel['relation']}" for rel in relations]
    result = "\n".join(summary)
    return result



def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True


def relation_filter(entity_id, steps, entity_name, pre_relations, pre_head, question, model, args):
    suggested = ','.join(extract_relations(steps))
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    
    sparql_relations_extract_tail = sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations + tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal

    combined_results = filtering(total_relations,model,question,suggested,2)  # Remove duplicates
    # print(combined_results)
    flag, retrieve_relations = select_relations(combined_results, entity_id, head_relations, tail_relations)
    return retrieve_relations  # format error or too small max_length



def retrieve_top_rels(total_relations, suggested, model, width=3):
    if isinstance(suggested, str):
        suggested_relations = suggested.split(',')
    else:
        suggested_relations = suggested

    scores = []
    for rel in total_relations:
        max_score = max([util.dot_score(model.encode(rel), model.encode(sug))[0].cpu().item() for sug in suggested_relations])
        scores.append(max_score)
    doc_score_pairs = sorted(list(zip(total_relations, scores)), key=lambda x: x[1], reverse=True)
    corpus_embeddings = model.encode(total_relations, convert_to_tensor=True, device='cuda')
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    query_embeddings = model.encode(suggested_relations, convert_to_tensor=True, device='cuda')
    query_embeddings = util.normalize_embeddings(query_embeddings)
    sim_matrix = util.dot_score(query_embeddings, corpus_embeddings)  # shape: (num_query, num_corpus)
    max_scores = torch.max(sim_matrix, dim=0).values  # Maximum score for each relation
    weights = torch.nn.functional.softmax(max_scores, dim=0)
    weights_np = weights.cpu().numpy()
    entropy = -np.sum(weights_np * np.log(weights_np + 1e-12))
    if len(total_relations) == 0 or np.log(len(total_relations)) == 0:
        entropy_norm = 0
    else:
        entropy_norm = entropy / np.log(len(total_relations))
    # adaptive width
    min_width, max_width = 3, 10  
    soft_width = int(min_width + (max_width - min_width) * entropy_norm)
    soft_width = max(min(soft_width, len(total_relations)), min_width)
    
    # top soft_width
    top_indices = np.argsort(-weights_np)[:soft_width]
    top_docs = [total_relations[i] for i in top_indices]
    
    return top_docs


def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # print(results["results"]["bindings"])
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]


def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return entity_id
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']
    


def select_relations(string, entity_id, head_relations, tail_relations):
    if isinstance(string,str):
        last_brace_l = string.rfind('[')
        last_brace_r = string.rfind(']')
        
        if last_brace_l < last_brace_r:
            string = string[last_brace_l:last_brace_r+1]
    
        relations=[]
        rel_list = eval(string.strip())
    elif isinstance(string,list):
        rel_list=string
        relations=[]
    for relation in rel_list:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": True})
        elif relation in tail_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": False})
    
    if not relations:
        return False, "No relations found"
    return True, relations

def filtering(total_relations,model,question,suggested,breadth=5):
    result = []
    if len(total_relations)>10:
        result += retrieve_top_rels(total_relations, question, model, width=breadth)
        if suggested:
            result += retrieve_top_rels(total_relations, suggested, model, width=2)
        combined_results = list(set(result))  # Remove duplicates
    else:
        result += retrieve_top_rels(total_relations, question, model, width=breadth)
        if suggested:
            result += retrieve_top_rels(total_relations, suggested, model, width=breadth)

        return total_relations
    return combined_results

def construct_relation_prune_prompt(question, steps, entity_name, total_relations, args):
    return extract_relation_prompt + question + '\nSuggested Relation: ' + ','.join(extract_relations(steps)) + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations)

def construct_entity_score_prompt(question, entity,relation, observation, entity_candidates):
    if observation:
        return prune_entity_prompt.format(question) +'\nSuggested Entites:'+observation+"\nTriplets: "+entity+ ', ' + relation+', ['+ "; ".join(entity_candidates) + ']'+'\nOutput: '
    else:
        return prune_entity_prompt.format(question)+'\nTriplets: '+entity+ ', ' + relation+', ['+ "; ".join(entity_candidates) + ']'+'\nOutput: '

def relation_search_prune(entity_id, steps, entity_name, pre_relations, pre_head, question,model, args):
    suggested=','.join(extract_relations(steps))
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    
    sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    if len(total_relations)>10:
        total_relations=filtering(total_relations,model,question,suggested,breadth=3)

    total_relations.sort()  # make sure the order in prompt is always equal

    prompt = construct_relation_prune_prompt(question, steps, entity_name, total_relations, args)
    result, token_num = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type,args.tokenizer,args.model, False, False)
    flag, retrieve_relations = select_relations(result, entity_id, head_relations, tail_relations) 

    if flag:
        return retrieve_relations, token_num
    else:
        result=filtering(total_relations,model,question,suggested,breadth=2)
        flag, retrieve_relations = select_relations(result, entity_id, head_relations, tail_relations)
        return retrieve_relations, token_num # format error or too small max_length


def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (relation, entity)
        entities = execurte_sparql(head_entities_extract)
    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity


def provide_triple(entity_candidates_id, relation):
    entity_candidates = []
    for entity_id in entity_candidates_id:
        if entity_id.startswith("m."):
            entity_candidates.append(id2entity_name_or_type(entity_id))
        else:
            entity_candidates.append(entity_id)

    if len(entity_candidates) <= 1:
        return entity_candidates, entity_candidates_id


    ent_id_dict = dict(sorted(zip(entity_candidates, entity_candidates_id)))
    entity_candidates, entity_candidates_id = list(ent_id_dict.keys()), list(ent_id_dict.values())
    return entity_candidates, entity_candidates_id

import random

def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    print(zipped)
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_zipped = [x for x in sorted_zipped if x[5] != 0]
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id, sorted_relations, sorted_candidates[:], sorted_topic_entities[:], sorted_head[:], sorted_scores[:]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:        
        new_sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
        # print(new_sorted_zipped)
        if len(new_sorted_zipped)>3:
            random.shuffle(new_sorted_zipped)
            new_sorted_zipped = new_sorted_zipped[:3]
        sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in new_sorted_zipped], [x[1] for x in new_sorted_zipped], [x[2] for x in new_sorted_zipped], [x[3] for x in new_sorted_zipped], [x[4] for x in new_sorted_zipped], [x[5] for x in new_sorted_zipped]
        entities_id, relations, candidates, topics, heads, scores = sorted_entities_id, sorted_relations, sorted_candidates[:], sorted_topic_entities[:], sorted_head[:], sorted_scores[:]
        merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
        filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))
    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads
    

def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head

def half_stop(question, question_string, subquestions, cluster_chain_of_entities, depth, call_num, all_t, start_time, args):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    call_num += 1
    answer, token_num = generate_answer(question, subquestions, cluster_chain_of_entities, args)

    for kk in token_num.keys():
        all_t[kk] += token_num[kk]

    save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type)


def generate_answer(question, subquestions, cluster_chain_of_entities, args): 
    prompt = answer_prompt + question 
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt
    result, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type,args.tokenizer,args.model, False)
    return result, token_num


def if_topic_non_retrieve(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def is_all_digits(lst):
    for s in lst:
        if not s.isdigit():
            return False
    return True
def entity_score(question,steps, entity_candidates_id, relation,entity, model,args):
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
    entity=id2entity_name_or_type(entity)
    observation=",".join(extract_triplets(steps))
    # if all_unknown_entity(entity_candidates):
    #     return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    # entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [1.0], entity_candidates, entity_candidates_id,''
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id,''
    if len(entity_candidates) > 10:
        if observation:
            entity_candidates=retrieve_top_rels(entity_candidates,observation, model, width=10)
        else:
            entity_candidates=retrieve_top_rels(entity_candidates,question, model, width=10)

    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    prompt = construct_entity_score_prompt(question, entity,relation,observation, entity_candidates)
    result, token_num = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type, args.tokenizer,args.model,False, False)
    print(prompt,result)
    if all(float(x) == 0.0 for x in clean_results(result, entity_candidates)):
        print("retry")
        prompt = construct_entity_score_prompt(question, entity,relation,'', entity_candidates)
        result, token_num = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type, args.tokenizer,args.model,False, False)
        print(prompt,result)
    return [float(x) * 1.0 for x in clean_results(result, entity_candidates)], entity_candidates, entity_candidates_id,token_num

def entity_condition_prune(question, total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, ent_rel_ent_dict, entid_name, name_entid, args, model):
    cur_call_time = 0
    cur_token = {'total': 0, 'input': 0, 'output': 0}

    new_ent_rel_ent_dict = {}
    no_prune = ['time', 'number', 'date']
    filter_entities_id, filter_tops, filter_relations, filter_candidates, filter_head = [], [], [], [], []
    for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela, e_list in sorted(r_e_dict.items()):
                if is_all_digits(e_list) or rela in no_prune or len(e_list) <= 1:
                    sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                    select_ent = sorted_e_list
                else:
                    if all(entid_name[item].startswith('m.') for item in e_list) and len(e_list) > 10:
                        e_list = random.sample(e_list, 10)

                    if len(e_list) > 20:
                        sorted_e_list = [entid_name[e_id] for e_id in e_list]
                        topn_entities, topn_scores = retrieve_top_docs(question, sorted_e_list, model, 20)
                        e_list = [name_entid[e_n] for e_n in topn_entities]
                        print('sentence:', topn_entities)

                    prompt = prune_entity_prompt + question +'\nTriples: '
                    sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                    prompt += entid_name[topic_e] + ' ' + rela + ' ' + str(sorted_e_list)

                    cur_call_time += 1
                    result, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, args.tokenizer,args.model, False, False)
                    for kk in token_num.keys():
                        cur_token[kk] += token_num[kk]

                    last_brace_l = result.rfind('[')
                    last_brace_r = result.rfind(']')
                    
                    if last_brace_l < last_brace_r:
                        result = result[last_brace_l:last_brace_r+1]
                    
                    try:
                        result = eval(result.strip())
                    except:
                        result = result.strip().strip("[").strip("]").split(', ')
                        result = [x.strip("'") for x in result]

                    select_ent = sorted(result)
                    select_ent = [x for x in select_ent if x in sorted_e_list]

                if len(select_ent) == 0 or all(x == '' for x in select_ent):
                    continue

                if topic_e not in new_ent_rel_ent_dict.keys():
                    new_ent_rel_ent_dict[topic_e] = {}
                if h_t not in new_ent_rel_ent_dict[topic_e].keys():
                    new_ent_rel_ent_dict[topic_e][h_t] = {}
                if rela not in new_ent_rel_ent_dict[topic_e][h_t].keys():
                    new_ent_rel_ent_dict[topic_e][h_t][rela] = []
                
                for ent in select_ent:
                    if ent in sorted_e_list:
                        new_ent_rel_ent_dict[topic_e][h_t][rela].append(name_entid[ent])
                        filter_tops.append(entid_name[topic_e])
                        filter_relations.append(rela)
                        filter_candidates.append(ent)
                        filter_entities_id.append(name_entid[ent])
                        if h_t == 'head':
                            filter_head.append(True)
                        else:
                            filter_head.append(False)


    if len(filter_entities_id) == 0:
        return False, [], [], [], [], new_ent_rel_ent_dict, cur_call_time, cur_token


    cluster_chain_of_entities = [[(filter_tops[i], filter_relations[i], filter_candidates[i]) for i in range(len(filter_candidates))]]
    return True, cluster_chain_of_entities, filter_entities_id, filter_relations, filter_head, new_ent_rel_ent_dict, cur_call_time, cur_token


def verify(question, steps, cluster_chain_of_entities, args):
    Verify_prompt = """Based on the knowledge triplets, analyze whether it is necessary to revise LLM's observation or add additional triplets in LLM's observation. The entity might have ID form such as m.xxxx. If all entities starts with 'm.', just ignore them and no need to revise. 
You are only allowed to use the information from knowledge triplets. DO NOT use your own knowledge.

Here are few examples:
LLM output:   
Thought 2: Now that I have a list of countries where Portuguese is spoken, I need to find out which of these countries had a child labor percentage of 1.8%.  
Action 2: Search[child labor percentage in Portugal, Brazil, Angola, Mozambique, Guinea-Bissau, East Timor, Equatorial Guinea, Cape Verde, São Tomé and Príncipe]  
Observation 2: (Brazil, unit.dated_percentage.rate, 1.8%), (Angola, unit.dated_percentage.rate, 1.5%), (Mozambique, unit.dated_percentage.rate, 2.0%), (Guinea-Bissau, unit.dated_percentage.rate, 2.3%), (East Timor, unit.dated_percentage.rate, 1.0%), (Equatorial Guinea, unit.dated_percentage.rate, 1.2%), (Cape Verde, unit.dated_percentage.rate, 1.4%), (São Tomé and Príncipe, unit.dated_percentage.rate, 1.6%), (Portugal, unit.dated_percentage.rate, 0.5%)  
Knowledge Triplets: (Mozambique, unit.dated_percentage.rate, 1.8%),(Brazil, unit.dated_percentage.rate, 1.8%)
Output: 
{
    "Revise": "Yes",
    "Reason": "Based on the triplets, the observations have factual errors that might lead to wrong answer.",
    "Revised Observation": (Brazil, unit.dated_percentage.rate, 2.0%), (Angola, unit.dated_percentage.rate, 1.5%), (Mozambique, unit.dated_percentage.rate, 1.8%), (Guinea-Bissau, unit.dated_percentage.rate, 2.3%), (East Timor, unit.dated_percentage.rate, 1.0%), (Equatorial Guinea, unit.dated_percentage.rate, 1.2%), (Cape Verde, unit.dated_percentage.rate, 1.4%), (São Tomé and Príncipe, unit.dated_percentage.rate, 1.6%), (Portugal, unit.dated_percentage.rate, 0.5%)  
}
LLM output:  
Thought 2: Now that I have identified that "Noble patria, tu hermosa bandera" is the national anthem of Venezuela, I need to find out what currency is used in that country.
Action 2: Search[currency of Venezuela]  
Observation 2: ('Venezuela', currency is, 'Venezuelan bolívar')
Knowledge Triplets: ("m.0h_1h3x","location.country.national_anthem","Costa Rica")
Output: 
{
    "Revise": "Yes",
    "Reason": "Based on the triplet, the observations have factual errors that might lead to wrong answer.",
    "Revised Observation":(Noble patria, tu hermosa bandera', national anthem of, Costa Rica)
}
LLM output:   
Thought 1: I need to know the artist of the "Country Nation World Tour"
Action 1: Search[artist of "Country Nation World Tour"]
Observation 1: （Country Nation World Tour, artist, Brad Paisley)

Knowledge Triplets: ('Country Nation World Tour','music.concert_tour.artist','Brad Paisley')
Output: 
{
    "Revise": "No",
    "Reason": "Based on the triplets, current observation is factually correct."
}

Now you need to directly output the results of the following question in the JSON format (must include "Revise" and "Reason", if 'yes' in revise, include "Revised Observation") without other information or notes.
Q: """
    prompt = Verify_prompt  + '\n Current step: ' + str(steps)

    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets:\n" + chain_prompt
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type,args.tokenizer,args.model, False)
    
    flag, reason, observation=extract_revise_and_observation(response)
    return flag, reason, observation,token_num

def verifying(steps, cluster_chain_of_entities,tokenizer,model):
    ob1=",".join(extract_triplets(steps))
    print(ob1)

    ob2="".join([', '.join([str(x) for x in chain]) for chain in cluster_chain_of_entities])
    ob2=str(ob2)
    
    input = ob1+ ' [SEP] ' + ob2
    print(input)
    encoded_input = tokenizer.encode(input, padding=True)
    prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
    predicted_label = torch.argmax(prediction, dim=1)
    reverse_input = ob2+ ' [SEP] ' + ob1
    encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
    reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)
    if 0 in predicted_label or 0 in reverse_predicted_label:
        flag = True
        return flag,ob2
    else:   
        flag = False
        return flag,''

def generate_plan(question,topic_entities,args):
    API_prompt='''
    You are an intelligent assistant tasked with answering the following question. Your job is to understand the question and plan all the necessary steps to solve it. Do not judge the question or give an unknown answer. Do NOT make too long (>10 steps) plans.
    You can only use the following two actions:
    (1) Search[Keyword]: To retrieve relative information based on the given question.  
    (2) Finish[Answer]: When the observations are sufficient to answer the question, return the final answer and finish the task.  
    
    Your response should only include the Thought, Action and Observation steps required to solve the question until the final answer is reached. Each step must strictly follow the format:
    Thought i:
    Action i:
    Observation i:
    The observation should be in the format of knowledge triplets. More than one triplet is allowed in observation.
    ### Here are the examples:
    Question:In which countries do the people speak Portuguese, where the child labor percentage was once 1.8?
    Thought 1: I need to identify the countries where Portuguese is spoken.  
    Action 1: Search[countries where Portuguese is spoken]  
    Observation 1: (Portuguese, spoken in, Portugal), (Portuguese, spoken in, Brazil), (Portuguese, spoken in, Angola), (Portuguese, spoken in, Mozambique), (Portuguese, spoken in, Guinea-Bissau), (Portuguese, spoken in, East Timor), (Portuguese, spoken in, Equatorial Guinea), (Portuguese, spoken in, Cape Verde), (Portuguese, spoken in, São Tomé and Príncipe)  
    
    Thought 2: Now that I have a list of countries where Portuguese is spoken, I need to find out which of these countries had a child labor percentage of 1.8%.  
    Action 2: Search[child labor percentage in Portugal, Brazil, Angola, Mozambique, Guinea-Bissau, East Timor, Equatorial Guinea, Cape Verde, São Tomé and Príncipe]  
    Observation 2: (Brazil, unit.dated_percentage.rate, 2.0%), (Angola, unit.dated_percentage.rate, 1.5%), (Mozambique, unit.dated_percentage.rate, 1.8%), (Guinea-Bissau, unit.dated_percentage.rate, 2.3%), (East Timor, unit.dated_percentage.rate, 1.0%), (Equatorial Guinea, unit.dated_percentage.rate, 1.2%), (Cape Verde, unit.dated_percentage.rate, 1.4%), (São Tomé and Príncipe, unit.dated_percentage.rate, 1.6%), (Portugal, unit.dated_percentage.rate, 0.5%)  
    
    Thought 3: The only country where Portuguese is spoken and had a child labor percentage of 1.8% is Mozambique.  
    Action 3: Finish[Mozambique]

    Question:Who held his governmental position from before Janaury 6, 2003 and was the 2009 Governor of Arizona?
    Thought 1: I need to find out who was the Governor of Arizona in 2009.  
    Action 1: Search[2009 Governor of Arizona]  
    Observation 1: (Janet Napolitano, served as, Governor of Arizona), (Janet Napolitano, served from, January 2003 to January 2009)  
    
    Thought 2: Now that I know Janet Napolitano was the Governor of Arizona in 2009, I need to check if she held her governmental position before January 6, 2003.  
    Action 2: Search[Janet Napolitano governmental positions before January 6, 2003]  
    Observation 2: (Janet Napolitano, served as, Attorney General of Arizona), (Janet Napolitano, served as, U.S. Attorney for Arizona)  
    
    Thought 3: Janet Napolitano held governmental positions as Attorney General of Arizona and U.S. Attorney for Arizona before January 6, 2003.  
    Action 3: Finish[Janet Napolitano]
    Follow these example and answer Question:{question}. Do not output other additional information. You can only end with Finish[] actions.
    
    '''
    prompt=API_prompt.format(question=question)
    if topic_entities:
        topic_entity=','.join(list(topic_entities.values()))
        prompt+=f'You should start with {topic_entity} in the first step.'          
                             
    plan,token_num=run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type,args.tokenizer,args.model,True,True)
    steps=extract_all_thoughts_and_actions(plan)
    return plan,steps,token_num

def update_plan(question,steps,args):
    API_prompt = '''
Complete the plan for the given question with the following steps:
Here is two examples:
Question: What movie with a prequel called "Escape to Witch Mountain" was Kim Richards in?
Thought 1: I need to identify the movie that has a prequel called "Escape to Witch Mountain" and features Kim Richards.  
Action 1: Search[Kim Richards movie "Escape to Witch Mountain"]  
Observation 1: (Kim Richards, starred in, "Escape to Witch Mountain"), ("Escape to Witch Mountain", has prequel, "Return from Witch Mountain")  
Thought 2: Now that I have identified that Kim Richards starred in "Escape to Witch Mountain," I need to confirm if there is another movie related to it that features her.  
Action 2: Search[Kim Richards filmography]  
Observation 2: (Kim Richards, starred in, "Return from Witch Mountain"), (Kim Richards, starred in, "Escape to Witch Mountain")  
Output:Thought 3: I have confirmed that Kim Richards starred in both "Escape to Witch Mountain" and its sequel "Return from Witch Mountain." Since the question asks for the movie with a prequel called "Escape to Witch Mountain," I can conclude that the answer is "Return from Witch Mountain."  
Action 3: Finish[Return from Witch Mountain]
Example 2:
Question:What currency is used in the country with Nobel Patria, tu hermosa as its national anthem?
Thought 1: I need to identify the country that has "Noble patria, tu hermosa bandera" as its national anthem.
Action 1: Search["Noble patria, tu hermosa bandera" national anthem country]
Observation 1: '("Noble patria, tu hermosa bandera", national anthem of, country associated with m.0h_1h3x)'
Thought 2: Now that I have identified the country associated with the national anthem "Noble patria, tu hermosa bandera," I need to find out what currency is used in that country.
Action 2: Search[currency of country associated with "Noble patria, tu hermosa bandera"]
Observation 2: ("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
Output:Thought 3: I have confirmed that the national anthem "Noble patria, tu hermosa bandera" belongs to Costa Rica. Now I need to state the currency used in Costa Rica.  
Action 3: Finish[Costa Rican colón]
Follow this example and continue current steps to answer initial question.  Use the same word in current observation and ensure you answer the question.
Question:
'''
    prompt = API_prompt+question+'\nSteps:'+format_steps(steps)+'Please revise given steps to complete the whole plan. \nOutput:'
    response,token_num=run_llm(prompt, args.temperature_reasoning ,args.max_length, args.opeani_api_keys, args.LLM_type,args.tokenizer,args.model,True,True)
    new_steps=extract_all_thoughts_and_actions(response)
    print(new_steps,steps)
    if 'Thought 1' not in response:
        plan=steps+new_steps
    else:
        plan=new_steps
    return response,plan,token_num



def update_plan_lookahead(question,steps,current_entity_relations_list,args):
    API_prompt='''
    You are provided with a reasoning plan for the following question and future relations as reference.Based on the given context, revise the plan as needed to correctly answer the question. 
    Your job is to understand the question and plan all the necessary steps to solve it. 
    You can use the following two actions:
    (1) Search[Keyword]: To retrieve relative information based on the given question.  
    (2) Finish[Answer]: When the observations are sufficient to answer the question, return the final answer extracted from the observation.  
    
    Your response should include all the Thought, Action and Observation steps required to solve the question until the final answer is reached. Each step must follow the format:
    Thought i:
    Action i:
    Observation i:
    The observation should be in the format of knowledge triplets. More than one triplet is allowed in observation.
    ### Example 1:
    Question: What movie with a prequel called "Escape to Witch Mountain" was Kim Richards in?
    Thought 1: I need to identify the movie that has a prequel called "Escape to Witch Mountain" and features Kim Richards.  
    Action 1: Search[Kim Richards movie "Escape to Witch Mountain"]  
    Observation 1: (Kim Richards, starred in, "Escape to Witch Mountain"), ("Escape to Witch Mountain", has prequel, "Return from Witch Mountain")  
    Thought 2: Now that I have identified that Kim Richards starred in "Escape to Witch Mountain," I need to confirm if there is another movie related to it that features her.  
    Action 2: Search[Kim Richards filmography]  
    Observation 2: (Kim Richards, starred in, "Return from Witch Mountain"), (Kim Richards, starred in, "Escape to Witch Mountain")  
    Thought 3: I have confirmed that Kim Richards starred in both "Escape to Witch Mountain" and its sequel "Return from Witch Mountain." Since the question asks for the movie with a prequel called "Escape to Witch Mountain," I can conclude that the answer is "Return from Witch Mountain."  
    Action 3: Finish[Return from Witch Mountain]
    ### Example 2:
    Question:What currency is used in the country with Nobel Patria, tu hermosa as its national anthem?
    Thought 1: I need to identify the country that has "Noble patria, tu hermosa bandera" as its national anthem.
    Action 1: Search["Noble patria, tu hermosa bandera" national anthem country]
    Observation 1: '("Noble patria, tu hermosa bandera", national anthem of, country associated with m.0h_1h3x)'
    Now you can observe:("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Please use this current observation to revise given steps and complete the whole plan.
    Output:Thought 2: Now that I have identified the country associated with the national anthem "Noble patria, tu hermosa bandera," I need to find out what currency is used in that country.
    Action 2: Search[currency of country associated with "Noble patria, tu hermosa bandera"]
    Observation 2: ("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Thought 3: I have confirmed that the national anthem "Noble patria, tu hermosa bandera" belongs to Costa Rica. So the answer to the question is Costa Rica.  
    Action 3: Finish[Costa Rican colón] 
    ###
    Here is the next step's information from KG for your to continue above steps.{info}
    Strictly follow the format in example and use above information to continue current steps below and answer initial question. Try to use exact words from observation and ensure you answer the question.
    Question:{question}
    {steps}
    '''
    prompt=API_prompt.format(question=question,steps=format_steps(steps),info=process_entity_relations(current_entity_relations_list))
    response,token_num=run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type,args.tokenizer,args.model,True,True)
    new_steps=extract_all_thoughts_and_actions(response)
    print(new_steps,steps)
    if 'Thought 1' not in response:
        plan=steps+new_steps
    else:
        plan=new_steps
    return response,plan,token_num

def update_plan_local(question,steps,observation,args):
    API_prompt = '''
    You are provided with a reasoning plan for the following question. Based on the given current observation, revise the plan as needed to correctly answer the question. You can also adjust previous steps based on how the observation aligns with or contradicts existing steps.The observations must strictly come from the KG triplets.
    
    Available actions:
    (1) Search[Keyword]: To retrieve relevant information based on the given question.  
    (2) Finish[Answer]: When the observations are sufficient to answer the question, return the final answer extracted from the observation.
    
    Your response should include all Thought, Action, and Observation steps required to solve the question until the final answer is reached. Each step must follow the format:
    Thought i:
    Action i:
    Observation i:
    
    The observation should be structured in knowledge triplets. Multiple triplets per observation are allowed.
    
    ### Example 1:
    Question: What movie with a prequel called \"Escape to Witch Mountain\" was Kim Richards in?
    Thought 1: I need to identify the movie with a prequel called \"Escape to Witch Mountain\" that features Kim Richards.  
    Action 1: Search[Kim Richards movie \"Escape to Witch Mountain\"]  
    Observation 1: (Kim Richards, starred in, \"Escape to Witch Mountain\"), (\"Escape to Witch Mountain\", has prequel, \"Return from Witch Mountain\")  
    Thought 2: Now that I have identified that Kim Richards starred in \"Escape to Witch Mountain,\" I should verify if she starred in the related film \"Return from Witch Mountain.\"  
    Action 2: Search[Kim Richards filmography]  
    Observation 2: (Kim Richards, starred in, \"Return from Witch Mountain\"), (Kim Richards, starred in, \"Escape to Witch Mountain\")  
    Thought 3: Kim Richards starred in both films. Since the question asks for the movie with a prequel called \"Escape to Witch Mountain,\" the answer is \"Return from Witch Mountain.\"  
    Action 3: Finish[Return from Witch Mountain]
    ### Example 2:
    Question:What currency is used in the country with Nobel Patria, tu hermosa as its national anthem?
    Thought 1: I need to identify the country that has "Noble patria, tu hermosa bandera" as its national anthem.
    Action 1: Search["Noble patria, tu hermosa bandera" national anthem country]
    Observation 1: '("Noble patria, tu hermosa bandera", national anthem of, country associated with m.0h_1h3x)'
    Now you can observe:("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Please use this current observation to revise given steps and complete the whole plan.
    Output:Thought 2: Now that I have identified the country associated with the national anthem "Noble patria, tu hermosa bandera," I need to find out what currency is used in that country.
    Action 2: Search[currency of country associated with "Noble patria, tu hermosa bandera"]
    Observation 2: ("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Thought 3: I have confirmed that the national anthem "Noble patria, tu hermosa bandera" belongs to Costa Rica. So the answer to the question is Costa Rica.  
    Action 3: Finish[Costa Rican colón] 
     
    Follow this example to revise or continue the current steps using exactly the current observation provided below. Ensure your final plan answers the original question correctly.
    
    Question: {question}
    Current steps: {steps}
    Current observation: {observation}
    
    Output:'''
    prompt = API_prompt.format(
        question=question,
        steps=format_steps(steps),
        observation=observation
    )
    # prompt = API_prompt+question+'\nSteps:'+format_steps(steps)+'\nCurrent observation:'+observation+'\nOutput:'
    response,token_num=run_llm(prompt, args.temperature_reasoning , args.max_length, args.opeani_api_keys, args.LLM_type,args.tokenizer,args.model,True,True)
    new_steps=extract_all_thoughts_and_actions(response)
    print(new_steps,steps)
    if 'Thought 1' not in response:
        plan=steps+new_steps
    else:
        plan=new_steps
    return response,plan,token_num



def update_plan_global(question, steps, cluster_chain_of_entities, args):
    API_prompt = '''
    You are provided with a reasoning plan for the following question and a set of knowledge graph (KG) triplets. The existing reasoning plan is severely incorrect or unreliable. Please revise the reasoning process completely from Thought 1.
    If the problem is conjunction, you are encouraged to utilize the triplets that have appeared in the Knowledge Graph (KG) to infer the earlier steps. The observations must strictly come from the KG triplets.
    
    (1) Search[Keyword]: To retrieve relative information based on the given question.  
    (2) Finish[Answer]: When the observations are sufficient to answer the question, return the final answer extracted from the observation.  
    
    Your response should include all the Thought, Action and Observation steps required to solve the question until the final answer is reached. Each step must follow the format:
    Thought i:
    Action i:
    Observation i:
    The observation should be in the format of knowledge triplets. More than one triplet is allowed in observation.
    Here is two examples:
    ### Example 1
    Question: What movie with a prequel called "Escape to Witch Mountain" was Kim Richards in?
    Thought 1: I need to identify the movie that has a prequel called "Escape to Witch Mountain" and features Kim Richards.  
    Action 1: Search[Kim Richards movie "Escape to Witch Mountain"]  
    Observation 1: (Kim Richards, starred in, "Escape to Witch Mountain"), ("Escape to Witch Mountain", has prequel, "Return from Witch Mountain")  
    Now you can observe:(Kim Richards, starred in, "Return from Witch Mountain"), (Kim Richards, starred in, "Escape to Witch Mountain") 
    Please use this current observation to revise given steps and complete the whole plan.
    Output:Thought 2: Now that I have identified that Kim Richards starred in "Escape to Witch Mountain," I need to confirm if there is another movie related to it that features her.  
    Action 2: Search[Kim Richards filmography]  
    Observation 2: (Kim Richards, starred in, "Return from Witch Mountain"), (Kim Richards, starred in, "Escape to Witch Mountain")  
    Thought 3: I have confirmed that Kim Richards starred in both "Escape to Witch Mountain" and its sequel "Return from Witch Mountain." Since the question asks for the movie with a prequel called "Escape to Witch Mountain," I can conclude that the answer is "Return from Witch Mountain."  
    Action 3: Finish[Return from Witch Mountain]
    ### Example 2:
    Question:What currency is used in the country with Nobel Patria, tu hermosa as its national anthem?
    Thought 1: I need to identify the country that has "Noble patria, tu hermosa bandera" as its national anthem.
    Action 1: Search["Noble patria, tu hermosa bandera" national anthem country]
    Observation 1: '("Noble patria, tu hermosa bandera", national anthem of, country associated with m.0h_1h3x)'
    Now you can observe:("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Please use this current observation to revise given steps and complete the whole plan.
    Output:Thought 2: Now that I have identified the country associated with the national anthem "Noble patria, tu hermosa bandera," I need to find out what currency is used in that country.
    Action 2: Search[currency of country associated with "Noble patria, tu hermosa bandera"]
    Observation 2: ("Noble patria, tu hermosa bandera", national anthem of, "Costa Rica"), ("Costa Rica", currency is, "Costa Rican colón")
    Thought 3: I have confirmed that the national anthem "Noble patria, tu hermosa bandera" belongs to Costa Rica. So the answer to the question is Costa Rica.  
    Action 3: Finish[Costa Rican colón] 
    ###
     Strictly follow the format in example without additional notation and use current KG retrieved triplets to revise whole steps below and answer initial question. Try to align each reasoning step with the provided KG triplets below. Ensure the final plan answers the original question correctly.
    Question:{question}
    Current plan:{steps}
    KG triplets:{KG}
    
    Output:'''
        
    prompt=API_prompt.format(question=question,steps=format_steps(steps),KG = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist]) )   
    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type,args.tokenizer,args.model, True, True)
    plan = extract_all_thoughts_and_actions(response)
    
    return response, plan, token_num