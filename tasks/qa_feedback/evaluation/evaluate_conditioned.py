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
                 feedback_types: List[List[str]],
                 dataloader: DataLoader) -> None:
        
        self.params = params
        self.nlf_cond = params['env']['nlf_cond']
        self.policy = policy
        self.score_model = score_model
        self.feedback_types = feedback_types
        self.dataloader = dataloader

        self.feedback_types = feedback_types
        self.best_feedbacks = [feedback[0] for feedback in feedback_types]
        if not self.nlf_cond: # Quark-based
            self.best_feedbacks_ids = self.policy.tokenizer.convert_tokens_to_ids(self.best_feedbacks)

    def add_feedback_to_prompt_input_ids(self,
                                         input_ids: torch.Tensor,
                                         attention_mask: torch.Tensor,
                                         tokenizer: AutoTokenizer,
                                         feedback: Optional[str] = None,
                                         best_feedback: Optional[bool] = True,
                                         feedback_quantiles: Optional[List[int]] = None,
                                         nlf_cond: Optional[bool] = True
                                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not nlf_cond: # Quark-based
            assert not feedback and not feedback_quantiles and best_feedback, "'nlf_cond'=False, i.e., Quark-based approach always conditions on best learned reward token during sampling. Do not specify other conditioning arguments."
            input_ids = torch.cat([input_ids.new([self.best_feedbacks_ids] * len(input_ids)), input_ids], dim=1)
            attention_mask = torch.cat([attention_mask.new([[1]] * len(attention_mask)), attention_mask], dim=1)
            return (input_ids, attention_mask)
        
        prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if feedback:
            assert best_feedback == False and not feedback_quantiles, "You can specify either conditioning on the best feedback, on specific feedback quantiles, or providing your own feedback, but not all."
            total_feedback = feedback

        else:
            total_feedback = ""
            if best_feedback:
                assert not feedback_quantiles, "Specify either conditioning on best feedback or on specific feedback quantiles."
                for idx, feedback in enumerate(self.best_feedbacks):
                    total_feedback += f"{feedback} " if idx != (len(self.best_feedbacks) - 1) else f"{feedback}." 
            else:
                assert feedback_quantiles and len(feedback_quantiles) == len(self.feedback_types), f"When 'best_feedback'=False, you need to specify a quantile index for each attribute you want to specify feedback (attribute 'feedback_quantiles'), i.e., {len(self.feedback_types)} "
                assert min(feedback_quantiles) >= 0 and max(feedback_quantiles) <= (len(self.feedback_types[0]) - 1), f"Invalid quantile indexs, they need to be within the range 0..{len(self.feedback_types[0]) - 1}"
                
                for idx, quantile_idx in enumerate(feedback_quantiles):
                    total_feedback += f"{self.feedback_types[idx][quantile_idx]} " if idx != (len(feedback_quantiles) - 1) else f"{self.feedback_types[idx][quantile_idx]}." 

        prompts = [(tokenizer.feedback_prefix + total_feedback + " " + tokenizer.prompt_prefix + prompt).strip() for prompt in prompts]

        input_dict = tokenizer(prompts, max_length=tokenizer.max_input_len, padding=True, truncation=True, return_tensors="pt")
        return (input_dict.input_ids, input_dict.attention_mask)

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

                input_ids_feedback, attention_mask = self.add_feedback_to_prompt_input_ids(
                    input_ids=input_ids, attention_mask=attention_mask,
                    tokenizer=self.policy.tokenizer,
                    best_feedback=True,
                    nlf_cond=self.nlf_cond
                )
                rollouts = self.policy.sample(prompts_input_ids=input_ids_feedback,
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
            f.write(f"Averaged Factuality score for all samples = {fact_scores_mean_over_num_samples:+.3f}\n")
            f.write(f"Averaged Factuality score for all sentences = {fact_scores_mean_over_num_sentences:+.3f}\n")
            f.write(f"Percentage of all sentences predicted as 'no error' = {fact_correct_ratio:+.3f}\n")
            f.write(f"Average generations lenght = {avg_generations_lens:+.3f}\n")
            f.write(f"Average RougeLSum = {avg_rouge_scores:+.3f}\n")
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

    nlf_cond = args['env']['nlf_cond']
    num_quantiles = args['env']['num_quantiles']
    num_attributes = args['env']['num_attributes']

    tokenizer = AutoTokenizer.from_pretrained(
        args['model']['policy_model']['base_model_ckpt'],
        model_max_length=args['env']['max_input_len']
    )
    tokenizer.max_input_len = args['env']['max_input_len']
    tokenizer.max_generated_len = args['env']['max_generated_len']
    if nlf_cond:
        tokenizer.feedback_prefix = "feedback: "
        tokenizer.prompt_prefix = "input: "
        feedback_types = [
            [
                "Most factual.",
                "Highly factual.",
                "Moderately factual.",
                "Slightly factual.",
                "Least factual."
             ],
        ]
        bad_words_ids = None
    else:
        tokenizer.feedback_prefix = ""
        tokenizer.prompt_prefix = ""
        feedback_types = [
            [f"_TREE_TOKEN_{str(attr)}_{str(quantile_idx)}"
                for quantile_idx in range(num_quantiles)]
        for attr in range(num_attributes)]

        # add special reward tokens to the tokenizer in case of Quark-like approach
        flattened_feedback_types = [feedback for feedback_type in feedback_types for feedback in feedback_type]
        tokenizer.add_tokens(flattened_feedback_types, special_tokens=True)
        bad_words_ids = [[tokenizer.convert_tokens_to_ids(flattened_feedback_type)] for flattened_feedback_type in flattened_feedback_types]
    
    policy = T5Policy(
        model_ckpt=args['model']['policy_model']['base_model_ckpt'],
        device=device,
        tokenizer=tokenizer,
        bad_words_ids=bad_words_ids
    )

    # resize token_embeddings associated to the newly added tokens in case of Quark-like approach
    if not nlf_cond:
        weights = policy.model.get_input_embeddings().weight.detach().cpu().numpy()
        mean_weights, std_weights = np.mean(weights, axis=0), np.std(weights, axis=0)
        new_inits = np.vstack([np.random.normal(loc=mean_weights, scale=std_weights) for _ in flattened_feedback_types])

        policy.model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            new_inits = torch.tensor(new_inits)
            policy.model.get_input_embeddings().weight[-len(flattened_feedback_types):, :] = new_inits

    model_name_or_path = args['model']['policy_model']['model_checkpoint_ckpt']
    if model_name_or_path is not None:
        checkpoint = torch.load(model_name_or_path, map_location='cpu')
        policy.model.load_state_dict(checkpoint['policy_model'])
        log.info("Model checkpoint loaded!")

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
        feedback_types=feedback_types,
        dataloader=dataloader
    )

    evaluator.eval(save_dir=args['logging']['save_dir'])

if __name__ == "__main__":
    main()