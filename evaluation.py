import os
from datasets import load_dataset

if __name__ == '__main__':
    dataset = load_dataset("cos_e", 'v1.0')
    print(dataset)
    test_set = dataset['validation']
    choices_list = []  # valid options
    for test in test_set:
        choices_list.append(['A: ' + test['choices'][0], 'B: ' + test['choices'][1], 'C: ' + test['choices'][2]])

    # few-shot evaluation
    non_zero_shot_paths = []
    search_string = "0shot"
    for dir in os.listdir("."):
        if os.path.isdir(dir):
            if search_string not in dir:
                non_zero_shot_paths.append(dir)

    for dir in non_zero_shot_paths:
        print(dir)
        with open(os.path.join(dir, 'responses.txt'), 'r') as r:
            r_list = r.read().split('-----------------------------')[:-1]
        with open(os.path.join(dir, 'prompts.txt'), 'r') as p:
            p_list = p.read().split('-----------------------------')[:-1]
        with open(os.path.join(dir, 'labels.txt'), 'r') as l:
            l_list = l.read().split('-----------------------------')[:-1]


        count = 0
        failure=0
        for i in range(len(l_list)):
            if choices_list[i][0] not in r_list[i] and choices_list[i][1] not in r_list[i] and choices_list[i][2] not in r_list[i]: # no valid option is predicted
                failure += 1
            if l_list[i].replace('\n', '') in r_list[i]:
                count += 1
        with open(os.path.join(dir, 'accuracy.txt'), 'w') as f:
            f.write(str(count/len(l_list)))
            f.write('\n')
            f.write(str(failure/len(l_list)))
    
    # zero-shot evaluation
    zero_shot_paths = []
    search_string = "0shot"
    for dir in os.listdir("."):
        if os.path.isdir(dir):
            if search_string in dir:
                zero_shot_paths.append(dir)

    for dir in zero_shot_paths:
        print(dir)
        with open(os.path.join(dir, 'responses.txt'), 'r') as r:
            r_list = r.read().split('-----------------------------')[:-1]
        with open(os.path.join(dir, 'prompts.txt'), 'r') as p:
            p_list = p.read().split('-----------------------------')[:-1]
        with open(os.path.join(dir, 'labels.txt'), 'r') as l:
            l_list = l.read().split('-----------------------------')[:-1]


        count = 0
        failure=0
        for i in range(len(l_list)):
            if r_list[i].replace('\n', '')[0] != 'A' and r_list[i].replace('\n', '')[0] != 'B' and r_list[i].replace('\n', '') != 'C': # first letter is not A/B/C
                failure += 1
            if l_list[i].replace('\n', '') == r_list[i].replace('\n', '')[0]:
                count += 1
        with open(os.path.join(dir, 'accuracy.txt'), 'w') as f:
            f.write(str(count/len(l_list)))
            f.write('\n')
            f.write(str(failure/len(l_list)))