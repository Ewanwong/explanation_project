import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, pipeline
import torch
import json
from datasets import load_dataset
import random
import os
import openai
import nltk
from tqdm import tqdm
import time
import math
import argparse
from accelerate import init_empty_weights
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_memory_mapping = {0: "25GB", 1: "25GB", 2: "25GB", 3: "25GB", 4: "25GB", 5: "25GB"} # to avoid memory error

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

class InContextLearning:
    def __init__(self, data, model_name='facebook/opt-66b',train_num=100, val_num=100, n_shot=5, use_instruction=True, greedy_search=False, use_explanation=True, shuffle_word_order=False, in_batch_explanations=False, answer_first=False, use_short_explanation=False, batch_size=20, use_oracle_explanation=False):
        self.train_num = train_num
        self.val_num = val_num
        self.test_num = len(data['validation']['id']) # use the validation set as test
        self.train_data = [data['train'][i] for i in range(self.train_num)]       
        self.val_data = [data['train'][i] for i in range(self.train_num, self.train_num+self.val_num)]
        self.test_data = data['validation']
        self.n_shot = n_shot 
        self.use_instruction = use_instruction
        self.greedy_search = greedy_search
        self.use_explanation = use_explanation
        self.shuffle_word_order = shuffle_word_order # scrambed explanations
        self.in_batch_explanations = in_batch_explanations # other item explanations
        self.answer_first = answer_first
        self.use_short_explanation = use_short_explanation
        self.batch_size = batch_size   
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True, device_map='auto', max_memory=max_memory_mapping)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left') 
        self.instruction = "Choose the correct answer to the question and provide an explanation to justify your answer.\n\n"
        self.example_ids = {'facebook/opt-125m': [71, 31, 60, 50, 35],
                            'facebook/opt-350m': [28, 22, 60, 87, 94],
                            'facebook/opt-1.3b': [1, 96, 93, 82, 64],
                            'facebook/opt-2.7b': [19, 14, 2, 56, 53]}
        self.use_oracle_explanation = use_oracle_explanation
        assert not use_oracle_explanation or not answer_first, 'oracle explanations can only be used when answer first is false'
        if use_oracle_explanation:
            self.instruction = "Choose the correct answer to the question based on the given explanation.\n\n"
    @staticmethod
    def set_random_seed(seed=1):
        random.seed(seed)
        set_seed(seed)
    
    def make_prompts(self, dir):
        # write prompts, label, explanation files
        prompt_path = os.path.join(dir, 'prompts.txt')
        label_path = os.path.join(dir, 'labels.txt')
        explanation_path = os.path.join(dir, 'explanations.txt')

        prompts = []
        correct_answers = []
        oracle_explanations = []
        if not self.greedy_search:
            for i in range(self.test_num):
                example_ids = self.get_random_subset() # randomly sample examples
                prompt, correct_answer, oracle_explanation = self.make_prompt(i, example_ids)
                prompts.append(prompt)
                correct_answers.append(correct_answer)
                oracle_explanations.append(oracle_explanation)
        else:
            for i in range(self.test_num):
                if self.model_name in self.example_ids.keys(): 
                    example_ids = self.example_ids[self.model_name][:self.n_shot]
                else:
                    example_ids = self.example_ids['facebook/opt-2.7b'][:self.n_shot] # for models larger than 2.7b, use examples found in 2.7b model
                prompt, correct_answer, oracle_explanation = self.make_prompt(i, example_ids)
                prompts.append(prompt)
                correct_answers.append(correct_answer)
                oracle_explanations.append(oracle_explanation)
        # write prompts, label, explanation files 
        with open(label_path, 'w', encoding='utf-8') as f:
            for label in correct_answers:
                f.write(label)
                f.write('\n')
                f.write('-----------------------------')
                f.write('\n')
        
        with open(prompt_path, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(prompt)
                f.write('\n')
                f.write('-----------------------------')
                f.write('\n')

        with open(explanation_path, 'w', encoding='utf-8') as f:
            for expl in oracle_explanations:
                f.write(expl)
                f.write('\n')
                f.write('-----------------------------')
                f.write('\n')

        return prompts, correct_answers, oracle_explanations

    def make_prompt(self, test_id, example_ids):
        assert len(example_ids) == self.n_shot, "Too many or too few examples."
        prompt = ''
        if self.use_instruction:
            assert self.instruction is not None, "Instruction must not be empty."
            prompt += self.instruction
        assert not (self.in_batch_explanations and len(example_ids) <= 1), 'Can only shuffle ids when n > 1'
        if len(example_ids) > 1 and self.in_batch_explanations:
            shuffled_ids = InContextLearning.shuffle_id_list(example_ids) # shuffle example ids, don't work for 1-shot 
        if len(example_ids) > 0:
            
            for i in range(len(example_ids)):
                id = example_ids[i]
                instance = self.train_data[id]
                question = 'Question: ' + instance['question'] + '\n'
                choices = 'Choice A: ' + instance['choices'][0] + '\n' + 'Choice B: ' + instance['choices'][1] + '\n' + 'Choice C: ' + instance['choices'][2] + '\n'
                answer = 'Answer: ' + ['A: ', 'B: ', 'C: '][instance['choices'].index(instance['answer'])] + instance['answer'] + '\n'
                if not self.use_explanation:
                    example = question + choices + answer + '\n'
                    prompt += example
                    continue
                if self.in_batch_explanations: # use other item's explanation
                    instance = self.train_data[shuffled_ids[i]]
                if self.use_short_explanation:
                    expl = instance['extractive_explanation']
                else:
                    expl = instance['abstractive_explanation']
                if self.shuffle_word_order: # use scrambled explanation
                    expl = InContextLearning.shuffle_explanation(expl)
                explanation = 'Explanation: ' + expl + '\n'
                if self.answer_first:
                    example = question + choices + answer + explanation + '\n'
                else:
                    example = question + choices + explanation + answer + '\n'
                prompt += example
        test = self.test_data[test_id]
        question = 'Question: ' + test['question'] + '\n'
        choices = 'Choice A: ' + test['choices'][0] + '\n' + 'Choice B: ' + test['choices'][1] + '\n' + 'Choice C: ' + test['choices'][2] + '\n'
        answer = 'Answer:'
        correct_answer = ['A: ', 'B: ', 'C: '][test['choices'].index(test['answer'])] + test['answer']
        if self.use_short_explanation:
            oracle_explanation = test['extractive_explanation']
        else:
            oracle_explanation = test['abstractive_explanation']
        if not self.use_explanation and not self.use_oracle_explanation:
            example = question + choices + answer
            prompt += example
            return prompt, correct_answer, oracle_explanation
        explanation = 'Explanation:'
        if self.use_oracle_explanation: # use oracle and answer first are not compatible
            explanation = explanation + ' ' + test['abstractive_explanation'] + '\n' + answer
        if self.answer_first:
            example = question + choices + answer
        else:
            example = question + choices + explanation
        prompt += example
        
        return prompt, correct_answer, oracle_explanation
    
    @staticmethod
    def shuffle_explanation(explanation): # shuffle word order
        words = nltk.word_tokenize(explanation)
        random.shuffle(words)
        return (' ').join(words)
    
    @staticmethod
    def shuffle_id_list(ids):
        # recursively return shuffled ids, such that no question pairs with original explanation
        num = len(ids)
        assert num > 1, "List can not be deranged."
        if num == 2:
            return [ids[1], ids[0]]
        last_ids = InContextLearning.shuffle_id_list(ids[:-1])
        
        i = random.randint(0, num-2)
        
        last_ids.append(last_ids[i])
        last_ids[i] = ids[num-1]
        return last_ids


    def get_random_subset(self):
        return random.sample(range(self.train_num), self.n_shot)

    def make_predictions(self, prompts, dir):
        
        response_path = os.path.join(dir, 'responses.txt')
        responses = []
        iter_num = math.ceil(self.test_num / self.batch_size)
        """
        with open(response_path, 'w', encoding='utf-8') as f:
            
            for prompt in tqdm(prompts):
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                generated_ids = self.model.generate(input_ids, do_sample=True, max_length=1024)
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                responses.append(response[0][len(prompt)+1:])
                f.write(response[0][len(prompt)+1:])
                f.write('\n')
                f.write('-----------------------------')
                f.write('\n')
        
        """
        
        with open(response_path, 'w', encoding='utf-8') as f:    
            for i in tqdm(range(iter_num)):
                if self.batch_size * (i + 1) >= len(prompts):
                    prompt = prompts[i*self.batch_size:]
                else:
                    prompt = prompts[i*self.batch_size:(i+1)*self.batch_size]  
                input_ids = self.tokenizer(prompt, padding=True, return_tensors="pt").to(device)
                # do generation
                with torch.no_grad():
                    generated_ids = self.model.generate(**input_ids, do_sample=True, max_new_tokens=60, top_p=0.9, temperature=0.9)
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                response_list = [response[i][len(prompt[i])+1:] for i in range(len(prompt))]
                responses += response_list
       
                for prediction in response_list:
                    f.write(prediction)
                    f.write('\n')
                    f.write('-----------------------------')
                    f.write('\n')

        # gpt-3 generation
        """
        openai.api_key = ""
        
        iter_num = math.ceil(self.test_num / self.batch_size)
        with open(response_path, 'w', encoding='utf-8') as f:
            
            for i in tqdm(range(iter_num)):
                if self.batch_size * (i + 1) >= len(prompts):
                    prompt = prompts[i*self.batch_size:]
                else:
                    prompt = prompts[i*self.batch_size:(i+1)*self.batch_size]          
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=60,
                    top_p=1,
                    frequency_penalty=0.5,
                    presence_penalty=0
                    )
                response_num = len(response['choices'])
                response_list = ['' for _ in range(response_num)]
                for choice in response.choices:
                    response_list[choice.index] = choice.text
                responses += response_list
       
                for prediction in response_list:
                    f.write(prediction)
                    f.write('\n')
                    f.write('-----------------------------')
                    f.write('\n')
         """    
        return responses

    @staticmethod
    def evaluate(answer_list, label_list):
        # find gold label in model output
        # only works with few-shot learning
        # run evaluation.py for more accurate evaluatio results
        assert len(answer_list) == len(label_list)
        correct = 0
        total = 0
        for i in range(len(label_list)):
            if label_list[i].replace('\n', '') in answer_list[i]:
                correct += 1
            total += 1
        
        return correct/total

    def run(self, seed, dir):
        self.set_random_seed(seed)
        if not os.path.exists(dir):
            os.mkdir(dir)
        
        prompts, correct_answers, oracle_explanations = self.make_prompts(dir)

        print("start making predictions")
        predictions = self.make_predictions(prompts, dir)
        accuracy = InContextLearning.evaluate(predictions, correct_answers)
        print(accuracy)
        #with open(os.path.join(dir, 'accuracy.txt'), 'w') as f:
        #    f.write(str(accuracy))
        return



if __name__ == '__main__':
    
    dataset = load_dataset("cos_e", 'v1.0')
    print(dataset)
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir', type=str, default='13b_default')
    parser.add_argument('--model_name', type=str, default='facebook/opt-13b')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_num', type=int, default=100)
    parser.add_argument('--val_num', type=int, default=100)
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--use_instruction', default='True')
    parser.add_argument('--greedy_search', default='False')
    parser.add_argument('--use_explanation', default='True')
    parser.add_argument('--shuffle_word_order', default='False')
    parser.add_argument('--in_batch_explanations', default='False')
    parser.add_argument('--answer_first', default='False')
    parser.add_argument('--use_short_explanation', default='False')
    parser.add_argument('--use_oracle_explanation', default='False')
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    # taking arguments from bash shell script
    if args.use_instruction == 'True':
        use_instruction = True
    elif args.use_instruction == 'False':
        use_instruction = False
    else:
        raise TypeError
    
    if args.greedy_search == 'True':
        greedy_search = True
    elif args.greedy_search == 'False':
        greedy_search = False
    else:
        raise TypeError

    if args.use_explanation == 'True':
        use_explanation = True
    elif args.use_explanation == 'False':
        use_explanation = False
    else:
        raise TypeError

    if args.shuffle_word_order == 'True':
        shuffle_word_order = True
    elif args.shuffle_word_order == 'False':
        shuffle_word_order = False
    else:
        raise TypeError
    
    if args.in_batch_explanations == 'True':
        in_batch_explanations = True
    elif args.in_batch_explanations == 'False':
        in_batch_explanations = False
    else:
        raise TypeError

    if args.answer_first == 'True':
        answer_first = True
    elif args.answer_first == 'False':
        answer_first = False
    else:
        raise TypeError

    if args.use_short_explanation == 'True':
        use_short_explanation = True
    elif args.use_short_explanation == 'False':
        use_short_explanation = False
    else:
        raise TypeError
    
    if args.use_oracle_explanation == 'True':
        use_oracle_explanation = True
    elif args.use_oracle_explanation == 'False':
        use_oracle_explanation = False
    else:
        raise TypeError

    model = InContextLearning(dataset, model_name=args.model_name, train_num=args.train_num, val_num=args.val_num, n_shot=args.n_shot, use_instruction=use_instruction, greedy_search=greedy_search, use_explanation=use_explanation, shuffle_word_order=shuffle_word_order, in_batch_explanations=in_batch_explanations, answer_first=answer_first, use_short_explanation=use_short_explanation, batch_size=args.batch_size, use_oracle_explanation=use_oracle_explanation)
    model.run(args.seed, args.dir)
    # post_process(args.dir)
        