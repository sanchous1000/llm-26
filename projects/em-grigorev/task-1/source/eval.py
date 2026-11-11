import yaml
from itertools import product

import pandas as pd
from tqdm.auto import tqdm

from call import ModelCall
from pathlib import Path


all_experiments = []


def read_questions(path="./configs/prompts/"):
    questions = []
    id = 0
    for file in sorted(Path(path).glob("*.txt")):
        questions.append({'text': file.read_text(encoding="utf-8"), "id": id})
        id += 1
    return questions


def read_params(path="./configs/models.yaml"):
    with open(path, encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    models_list = configs["models_list"]
    params = configs["params"]

    return models_list, params


def eval_one_model(model_name, product_params, questions, param_names):
    mc = ModelCall(model_name=model_name)

    # Один tqdm на все комбинации параметров
    params_bar = tqdm(
        product_params,
        desc=f"⚙️ {model_name}",
        unit="param",
        leave=False,
    )

    for iter_params in params_bar:

        current_params = dict(zip(param_names, iter_params))

        params_bar.set_postfix(
            **{
                key: value
                for key, value in current_params.items()
            }
        )

        for prompt in tqdm(
            questions,
            desc="  📝 prompts",
            unit="q",
            leave=False,
        ):
            ans = mc.call(
                prompt=prompt['text'],
                **current_params,
            )

            all_experiments.append({
                "model_name": model_name,
                "ans": ans,
                "params": current_params.copy(),
                "prompt": prompt['text'],
                "prompt_id": prompt['id']
            })


def iter_over_models(
    models_list,
    product_params,
    questions,
    param_names,
):
    models_bar = tqdm(
        models_list,
        desc="🤖 Models",
        unit="model",
    )

    for model_name in models_bar:

        models_bar.set_postfix(
            model=model_name
        )

        eval_one_model(
            model_name=model_name,
            product_params=product_params,
            questions=questions,
            param_names=param_names,
        )


# ============================================================
# Main
# ============================================================

models_list, params = read_params()

param_names = list(params.keys())

product_params = list(
    product(
        *[params[name] for name in param_names]
    )
)

questions = read_questions()
print(f"Models:      {len(models_list)}")
print(f"Parameters:  {len(product_params)}")
print(f"Questions:   {len(questions)}")
print(
    f"Total calls: "
    f"{len(models_list) * len(product_params) * len(questions)}"
)
print()

iter_over_models(
    models_list=models_list,
    product_params=product_params,
    questions=questions,
    param_names=param_names,
)

pd.DataFrame(all_experiments).to_parquet(
    "report.parquet"
)