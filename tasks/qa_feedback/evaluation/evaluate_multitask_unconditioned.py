from ctgnlf.datasets_and_collators import PromptCollator, PromptDataset
from ctgnlf.policy import T5Policy
from ctgnlf.utils import set_seed, ensure_dir
from ctgnlf.reward import MyFactualityRewardModel, MyRelevancyRewardModel, MyCompletenessRewardModel, MyFineGrainedRewardModel

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

        eval_output = self.score_model.get_metrics(
            prompt_texts=prompts, 
            generated_texts=responses, 
            batch_size=self.params['reward']['batch_size'],
            references=references
        )
        # RELEVANCY
        n_sub_sentences = eval_output["n_sub_sentences"]
        n_corrects_rel = eval_output["n_corrects_rel"]
        rel_rewards = eval_output["rel_rewards"]

        rel_scores_mean_over_num_samples = np.mean(rel_rewards) # averaged output score for all dev_set samples 
        rel_scores_mean_over_num_sentences = np.sum(rel_rewards) / np.sum(n_sub_sentences) # averaged output score for all dev_set senteces
        rel_correct_ratio = np.sum(n_corrects_rel) / np.sum(n_sub_sentences) # percentage of al sentences in the dev_set predicted as "no error"

        # FACTUALITY
        n_sentences = eval_output["n_sentences"]
        n_corrects_fact = eval_output["n_corrects_fact"]
        fact_rewards = eval_output["fact_rewards"]

        fact_scores_mean_over_num_samples = np.mean(fact_rewards) # averaged output score for all dev_set samples 
        fact_scores_mean_over_num_sentences = np.sum(fact_rewards) / np.sum(n_sentences) # averaged output score for all dev_set senteces
        fact_correct_ratio = np.sum(n_corrects_fact) / np.sum(n_sentences) # percentage of al sentences in the dev_set predicted as "no error"
        
        # COMPLETENESS
        comp_rewards = eval_output["comp_rewards"]

        comp_scores_mean_over_num_samples = np.mean(comp_rewards) # averaged output score for all dev_set samples 

        # OTHERS
        generations_lens = eval_output["generations_lens"]
        rouge_scores = eval_output["rouge_scores"]

        avg_generations_lens = np.mean(generations_lens)
        avg_rouge_scores = np.mean(rouge_scores)

        print(f"Average generations lenght = {avg_generations_lens:+.2f}")
        print(f"Average RougeLSum = {avg_rouge_scores:+.2f}")

        data_split = args['data']['data_path'].split("/")[-1].split(".")[0]
        with open(save_dir, 'a') as f:
            f.write(f"{data_split}:\n")
            f.write(f"Averaged Relevancy score for all samples = {rel_scores_mean_over_num_samples:+.3f}\n")
            f.write(f"Averaged Relevancy score for all sentences = {rel_scores_mean_over_num_sentences:+.3f}\n")
            f.write(f"(rel) Percentage of all subsentences predicted as 'no error' = {rel_correct_ratio:+.3f}\n")
            f.write(f"Averaged Factuality score for all samples = {fact_scores_mean_over_num_samples:+.3f}\n")
            f.write(f"Averaged Factuality score for all sentences = {fact_scores_mean_over_num_sentences:+.3f}\n")
            f.write(f"(fact) Percentage of all sentences predicted as 'no error' = {fact_correct_ratio:+.3f}\n")
            f.write(f"Averaged Completeness score for all samples = {comp_scores_mean_over_num_samples:+.3f}\n")
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

    model_name_or_path = args['model']['policy_model']['model_checkpoint_ckpt']
    if model_name_or_path is not None:
        checkpoint = torch.load(model_name_or_path, map_location='cpu')
        policy.model.load_state_dict(checkpoint)
        log.info("Model checkpoint loaded!")

    # initialize reward model and data pool
    relevancy_rm = MyRelevancyRewardModel(
        policy_tokenizer=tokenizer,
        reward_model_name_or_path=args['reward']['relevancy_model']['ckpt'],
        device=device,
        positive_reward=args['reward']['relevancy_model']['positive_reward'],
        negative_reward=args['reward']['relevancy_model']['negative_reward']
    )

    factuality_rm = MyFactualityRewardModel(
        policy_tokenizer=tokenizer,
        reward_model_name_or_path=args['reward']['factuality_model']['ckpt'],
        device=device,
        positive_reward=args['reward']['factuality_model']['positive_reward'],
        negative_reward=args['reward']['factuality_model']['negative_reward']
    )

    completeness_rm = MyCompletenessRewardModel(
        policy_tokenizer=tokenizer,
        reward_model_name_or_path=args['reward']['completeness_model']['ckpt'],
        device=device,
        mean=args['reward']['completeness_model']['mean'],
        std=args['reward']['completeness_model']['std'],
        bias=args['reward']['completeness_model']['bias'],
        scale=args['reward']['completeness_model']['scale']
    )

    reward_model = MyFineGrainedRewardModel(relevancy_rm, factuality_rm, completeness_rm)

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