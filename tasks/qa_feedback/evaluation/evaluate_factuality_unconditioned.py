from ctgnlf.datasets_and_collators import PromptCollator, PromptDataset
from ctgnlf.policy import T5Policy
from ctgnlf.utils import set_seed, ensure_dir
from ctgnlf.reward import MyFactualityRewardModel

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import logging 
import argparse
import os
import yaml
from tqdm import tqdm
from typing import List, Optional, Tuple, Union
from pathlib import Path

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO")) # log levels, from least severe to most severe, are: DEBUG, INFO, WARNING, ERROR, and CRITICAL.
log = logging.getLogger(__name__)

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)

class Evaluator:
    def __init__(self,
                 params: dict,
                 policy: T5Policy,
                 score_model: MyFactualityRewardModel,
                 dataloader: DataLoader) -> None:
        
        self.params = params
        self.policy = policy
        self.score_model = score_model
        self.dataloader = dataloader

    def decode(self, 
               tokenizer: AutoTokenizer,
               query_input_ids: torch.Tensor,
               response_input_ids: Optional[torch.Tensor] = None
               ) -> Union[List[str], Tuple[List[str], List[str]]]:
        
        query = tokenizer.batch_decode(query_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if response_input_ids is None:
            return query
        
        response = tokenizer.batch_decode(response_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return (query, response)

    def eval(self, save_dir: Union[str, os.PathLike]):

        save_dir = Path(save_dir) / "eval_results.txt"
        log.info("Evaluating ...")

        prompts, responses = [], []
        references = []
        for i, batch in enumerate(tqdm(self.dataloader, desc='Sampling from current policy')):
            with torch.no_grad():
                input_ids, attention_mask = batch["inputs"]
                references.extend(batch["references"])

                rollouts = self.policy.sample(prompts_input_ids=input_ids,
                                            prompts_attention_mask=attention_mask,
                                            do_sample=self.params['model']['policy_model']['eval_generation_kwargs']['do_sample'],
                )
                response = rollouts["generated_text"]
                prompt = self.decode(tokenizer=self.policy.tokenizer, query_input_ids=input_ids)

                prompts.extend(prompt)
                responses.extend(response)

        eval_output = self.score_model.get_full_reward_batch(
            prompt_texts=prompts, 
            generated_texts=responses, 
            batch_size=self.params['reward']['factuality_model']['batch_size'],
            references=references
        )

        n_sentences = eval_output["n_sentences"]
        n_corrects = eval_output["n_corrects"]
        pooled_rewards = eval_output["pooled_rewards"]
        generations_lens = eval_output["generations_lens"]
        rouge_scores = eval_output["rouge_scores"]

        fact_scores_mean_over_num_samples = np.mean(pooled_rewards) # averaged output score for all dev_set samples 
        fact_scores_mean_over_num_sentences = np.sum(pooled_rewards) / np.sum(n_sentences) # averaged output score for all dev_set senteces
        fact_correct_ratio = np.sum(n_corrects) / np.sum(n_sentences) # percentage of al sentences in the dev_set predicted as "no error"
        avg_generations_lens = np.mean(generations_lens)
        avg_rouge_scores = np.mean(rouge_scores)

        data_split = args['data']['data_path'].split("/")[-1].split(".")[0]
        with open(save_dir, 'a') as f:
            f.write(f"{data_split}:\n")
            f.write(f"Averaged Factuality score for all samples = {fact_scores_mean_over_num_samples:+.2f}\n")
            f.write(f"Averaged Factuality score for all sentences = {fact_scores_mean_over_num_sentences:+.2f}\n")
            f.write(f"Percentage of all sentences predicted as 'no error' = {fact_correct_ratio:+.2f}\n")
            f.write(f"Average generations lenght = {avg_generations_lens:+.2f}\n")
            f.write(f"Average RougeLSum = {avg_rouge_scores:+.2f}\n")
            f.write("\n")

def main():
    set_seed(
        seed=args['eval']['seed'], 
        cuda_deterministic=args['eval']['cuda_deterministic'])

    num_gpus = torch.cuda.device_count()
    log.info(f'Detected {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    log.info(f"Writing to output directory: {args['logging']['save_dir']}")
    ensure_dir(args['logging']['save_dir'])

    tokenizer = AutoTokenizer.from_pretrained(
        args['model']['policy_model']['base_model_ckpt'],
        model_max_length=args['env']['max_input_len']
    )
    tokenizer.max_input_len = args['env']['max_input_len']
    tokenizer.max_generated_len = args['env']['max_generated_len']
    
    policy = T5Policy(
        model_ckpt=args['model']['policy_model']['base_model_ckpt'],
        device=device,
        tokenizer=tokenizer,
    )

    # initialize reward model and data pool
    reward_model = MyFactualityRewardModel(
        policy_tokenizer=tokenizer,
        reward_model_name_or_path=args['reward']['factuality_model']['ckpt'],
        device=device,
        factuality_positive_reward=args['reward']['factuality_model']['positive_reward'],
        factuality_negative_reward=args['reward']['factuality_model']['negative_reward']
    )

    prompt_collator = PromptCollator(tokenizer=tokenizer)
    dataset = PromptDataset(path=args['data']['data_path'])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args['eval']['sampling_batch_size_per_card'],
        shuffle=False,
        drop_last=False,
        collate_fn=prompt_collator
    )

    evaluator = Evaluator(
        params=args,
        policy=policy,
        score_model=reward_model,
        dataloader=dataloader
    )

    evaluator.eval(save_dir=args['logging']['save_dir'])

if __name__ == "__main__":
    main()