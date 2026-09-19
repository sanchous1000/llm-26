import yaml
from itertools import product
import pandas as pd

from call import ModelCall

all_experiments = []

def read_questions(path='./configs/questions.yaml'):
    with open(path) as f:
        questions = yaml.safe_load(f)
    return questions

def read_params(path='./configs/models.yaml'):
    with open(path) as f:
        configs = yaml.safe_load(f)
    models_list, params = configs['models_list'], configs['params']
    return models_list, params

def eval_one_model(model_name):
    mc = ModelCall(model_name=model_name)
    for iter_params in product_params:
        temperature, repeat_penalty = iter_params
        for prompt in questions:
            ans = mc.call(prompt=prompt, temperature=temperature, repeat_penalty=repeat_penalty)

            all_experiments.append({'model_name': model_name, "ans": ans, "params": {list(params.keys())[i]: iter_params[i] for i in range(len(iter_params))}, "prompt": prompt})

def iter_over_models(models_list):

    for model_name in models_list:
        eval_one_model(model_name)

models_list, params = read_params()
product_params = [*product(*[params[i] for i in params])]
print(product_params)
questions = read_questions()


iter_over_models(models_list)
pd.DataFrame(all_experiments).to_parquet('report.parquet')

    