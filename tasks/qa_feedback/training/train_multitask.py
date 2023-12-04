import logging
import os
import argparse
import yaml
import json
from datetime import datetime
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
from pathlib import Path
import time
import sys
import heapq
import shutil

from transformers import AutoTokenizer, get_scheduler
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb

from ctgnlf.utils import set_seed, ensure_dir, ceil_div, reduce_sum, reduce_mean, WANDB_API_KEY
from ctgnlf.policy import T5Policy
from ctgnlf.datasets_and_collators import PromptDataset, PromptCollator, SequenceWithFeedbackDataset, SequenceWithFeedbackCollator
from ctgnlf.data_pool import DataPool
from ctgnlf.reward import MyFineGrainedRewardModel, MyRelevancyRewardModel, MyFactualityRewardModel, MyCompletenessRewardModel

# load parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

# load yaml file
with open(args.config) as f:
    args = yaml.safe_load(f)

# Not working?
# logging.basicConfig(steam=sys.stdout, level=logging.INFO) # log levels, from least severe to most severe, are: DEBUG, INFO, WARNING, ERROR, and CRITICAL.
# log = logging.getLogger(__name__)

class ConditionOnFeedbackTrainer:
    def __init__(self,
                 params: dict,
                 policy: T5Policy,
                 ref_policy: T5Policy,
                 score_model: MyFineGrainedRewardModel,
                 data_pool: DataPool,
                 feedback_types: List[List[str]],
                 train_dataloader: DataLoader,
                 dev_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LambdaLR
                 ) -> None:
        
        self.params = params

        self.nlf_cond = params['env']['nlf_cond']
        self.num_quantiles = params['env']['num_quantiles']
        self.num_attributes = params['env']['num_attributes']

        self.policy = policy
        self.ref_policy = ref_policy
        self.score_model = score_model
        self.data_pool = data_pool
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

        self.feedback_types = feedback_types
        self.best_feedbacks = [feedback[0] for feedback in feedback_types]
        if not self.nlf_cond: # Quark-based
            self.best_feedbacks_ids = self.policy.tokenizer.convert_tokens_to_ids(self.best_feedbacks)

        self.sample_dataloader, self.sampler = None, None
        self.seq_collator = SequenceWithFeedbackCollator(tokenizer=policy.tokenizer)

        self.top_models = []  # List to store top models
        self.top_models_limit = 5

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
            attention_mask = torch.cat([attention_mask.new([[1]*self.num_attributes] * len(attention_mask)), attention_mask], dim=1)
            return (input_ids, attention_mask)
        
        prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if feedback:
            assert best_feedback == False and not feedback_quantiles, "You can specify either conditioning on the best feedback, on specific feedback quantiles, or providing your own feedback, but not all."
            total_feedback = feedback

        else:
            total_feedback = ""
            if best_feedback:
                assert not feedback_quantiles, "Specify either conditioning on best feedback or on specific feedback quantiles."
                for idx, feedback in enumerate(self.best_feedbacks): # e.g., best_feedbacks = ["Most relevant.", "Most factual.", "Most complete."]
                    total_feedback += f"{feedback} " if idx != (self.num_attributes - 1) else f"{feedback}" 
            else:
                assert feedback_quantiles and len(feedback_quantiles) == self.num_attributes, f"When 'best_feedback'=False, you need to specify a quantile index for each attribute you want to specify feedback (attribute 'feedback_quantiles'), i.e., {self.num_attributes} "
                assert min(feedback_quantiles) >= 0 and max(feedback_quantiles) <= (self.num_quantiles - 1), f"Invalid quantile indexs, they need to be within the range 0..{self.num_quantiles - 1}"
                
                for idx, quantile_idx in enumerate(feedback_quantiles): # e.g., feedback_quantiles = [0, 0, 1] -> total_feedback = "Most relevant. Most factual. Highly complete."
                    total_feedback += f"{self.feedback_types[idx][quantile_idx]} " if idx != (self.num_attributes - 1) else f"{self.feedback_types[idx][quantile_idx]}." 

        prompts = [(tokenizer.feedback_prefix + total_feedback + " " + tokenizer.prompt_prefix + prompt).strip() for prompt in prompts]

        input_dict = tokenizer(prompts, max_length=tokenizer.max_input_len, padding=True, truncation=True, return_tensors="pt")
        return (input_dict.input_ids, input_dict.attention_mask)
    
    def remove_any_feedback_from_prompt_input_ids(self,
                                                  input_ids: torch.Tensor,
                                                  attention_mask: torch.Tensor,
                                                  tokenizer: AutoTokenizer,
                                                  nlf_cond: Optional[bool] = True
                                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if not nlf_cond: # Quark-based
            input_ids = input_ids[:, self.num_attributes:]
            attention_mask = attention_mask[:, self.num_attributes:]
            return (input_ids, attention_mask)
        
        prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        prompts = [prompt.split("input: ", 1)[1] for prompt in prompts]
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
    
    def sample(self, step_num):
        if step_num % self.params['env']['sample_interval'] != 0:
            return
        
        print(f"[step {step_num}] Sampling ...")

        prompts, responses, prompts_feedback = [], [], []
        for i, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader), desc='Sampling from current policy')):
            input_ids, attention_mask = batch["inputs"]

            # in the first sampling phase we sample from the reference policy without conditioning on any feedback / quantile reward tokens
            if step_num == 0:
                rollouts = self.ref_policy.sample(prompts_input_ids=input_ids,
                                                  prompts_attention_mask=attention_mask,
                                                  do_sample=self.params['model']['policy_model']['train_generation_kwargs']['do_sample'],
                                                  top_k=self.params['model']['policy_model']['train_generation_kwargs']['top_k'],
                                                  top_p=self.params['model']['policy_model']['train_generation_kwargs']['top_p'],
                                                  temperature=self.params['model']['policy_model']['train_generation_kwargs']['temperature'])
                prompt, response = rollouts["prompts_text"], rollouts["generated_text"]
                prompt_feedback = ["" for _ in range(len(prompt))]
            # otherwise, sample from the current policy conditioning on the best feedback / quantile reward tokens
            else:
                # preprend 'feedback: {best feedback} . input: ' (for ctg-nlf) / '{best quantile reward tokens}' (for Quark) to the prompt 
                input_ids_feedback, attention_mask = self.add_feedback_to_prompt_input_ids(
                    input_ids=input_ids, attention_mask=attention_mask,
                    tokenizer=self.policy.tokenizer,
                    best_feedback=True,
                    nlf_cond=self.nlf_cond
                )
                rollouts = self.policy.sample(prompts_input_ids=input_ids_feedback,
                                              prompts_attention_mask=attention_mask,
                                              do_sample=self.params['model']['policy_model']['train_generation_kwargs']['do_sample'],
                                              top_k=self.params['model']['policy_model']['train_generation_kwargs']['top_k'],
                                              top_p=self.params['model']['policy_model']['train_generation_kwargs']['top_p'],
                                              temperature=self.params['model']['policy_model']['train_generation_kwargs']['temperature'])
                response = rollouts["generated_text"]
                prompt = self.decode(tokenizer=self.policy.tokenizer, query_input_ids=input_ids)
                prompt_feedback = self.policy.tokenizer.batch_decode(input_ids_feedback, skip_special_tokens=False)

            prompts.extend(prompt)
            responses.extend(response)
            prompts_feedback.extend(prompt_feedback)

        # scores are computed on prompts without feedback
        scores = self.score_model.get_reward_batch(prompt_texts=prompts, generated_texts=responses, batch_size=self.params['reward']['batch_size'])
        # data_pool also receives prompts without feedback, as it orders the data points according to 
        # their scores and then assigns feedback to them
        self.data_pool.add(prompts=prompts, responses=responses, scores=scores)

        # save tuples of (prompt_feedback, promp, response, score) in reward_file
        reward_file = Path(self.params['reward_dir']) / f"multitask_rewards_{step_num}.json"
        with reward_file.open('a') as f:
            for idx, (prompt_feedback_data, prompt_data, response_data) in enumerate(zip(prompts_feedback, prompts, responses)):
                response_dict = {
                    'prompt_feedback': prompt_feedback_data,
                    'prompt': prompt_data,
                    'response': response_data,
                    'scores': [scores[attr][idx] for attr in range(self.num_attributes)] # i.e., for each sample [rel, fact, comp]
                }
                json.dump(response_dict, f)
                f.write('\n')

        sample_dataset = SequenceWithFeedbackDataset(data_pool=self.data_pool)
        self.sample_dataloader = DataLoader(
            dataset=sample_dataset,
            batch_size=self.params['train']['training_batch_size_per_card'],
            shuffle=True,
            drop_last=True,
            collate_fn=self.seq_collator
        )
        self.sampler = iter(self.sample_dataloader)

    def step(self, step_num):
        step_started_at = time.time()
        self.policy.model.eval()
        self.sample(step_num)
        self.policy.model.train()

        try:
            batch = next(self.sampler)
            assert len(batch[0]) == self.params['train']['training_batch_size_per_card'], 'insufficent batch'

        except (StopIteration, AssertionError):
            self.sampler = iter(self.sample_dataloader)  # reset iteration to the beginning of data
            batch = next(self.sampler)

        self.optimizer.zero_grad()
        loss, stats = self.loss(step_num, *batch)
        loss.backward()

        if self.params['train']['clip_grad']:
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.params['train']['max_grad_norm'])

        self.optimizer.step()
        self.scheduler.step()

        step_time = time.time() - step_started_at
        eps_per_second = float(self.params['train']['training_batch_size_per_card']) / step_time
        print(f"[step {step_num}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")     

        # --- EVALUATION ---
        self.policy.model.eval()
        eval_metric = self.eval(step_num) # eval_metric returned as the avg score between all RMs on all dev samples

       # --- LOGGING ---
        if self.params['logging']['wandb_log']:
            for metric in ['kl', 'entropy']:
                wandb.log({f'Objective/{metric}': stats[f'objective/{metric}']}, step=step_num)

            for metric in ['lm', 'kl', 'entropy', 'total']:
                wandb.log({f'Loss/{metric}': stats[f'loss/{metric}']}, step=step_num)

            wandb.log({f'Params/lr': self.optimizer.param_groups[0]['lr']}, step=step_num)

        # --- SAVING ---
        # Save the top models
        self.save(step_num, eval_metric)

    
    def loss(self, step_num, query_input_ids, query_mask, response_input_ids, response_mask):
        # query_input_ids, query_mask have already NLF / Reward quantiles tokens prepended
        outputs = self.policy.forward_pass(query_input_ids, query_mask, response_input_ids, response_mask)
        lm_loss, logprobs, entropy, logits = outputs['lm_loss'], outputs['generated_logprobs'], outputs['generated_entropy'], outputs['generated_logits']
        if not self.nlf_cond: # Quark-based
            flattened_feedback_types = [feedback for feedback_type in self.feedback_types for feedback in feedback_type]
            logits = logits[:, :, :-len(flattened_feedback_types)]
        masks = response_mask.to(self.policy.device)

        with torch.no_grad():
            query_input_ids, query_mask = self.remove_any_feedback_from_prompt_input_ids(input_ids=query_input_ids, 
                                                                                         attention_mask=query_mask, 
                                                                                         tokenizer=self.policy.tokenizer,
                                                                                         nlf_cond=self.nlf_cond)
            ref_outputs = self.ref_policy.forward_pass(query_input_ids, query_mask, response_input_ids, response_mask)
            ref_logprobs, ref_logits = ref_outputs['generated_logprobs'], ref_outputs['generated_logits']

        kl = torch.sum(self.kl_loss(F.log_softmax(logits, dim=-1), F.softmax(ref_logits, dim=-1)), dim=-1)
        loss = reduce_mean(lm_loss + self.params['env']['kl_coef']*kl - self.params['env']['entropy_coef']*entropy, masks)

        data = {'logprobs': logprobs, 'ref_logprobs': ref_logprobs, 'masks': masks,
                'logits': logits, 'ref_logits': ref_logits,
                'lm_loss': reduce_mean(lm_loss, masks), 'kl_loss': reduce_mean(kl, masks),
                'entropy': reduce_mean(entropy, masks), 'total_loss': loss}
        stats = self.record_step_stats(data)

        queries, responses = self.decode(self.policy.tokenizer, query_input_ids, response_input_ids)
        self.print_samples(queries=queries, responses=responses, lm_loss=reduce_mean(lm_loss, masks, axis=1),
                           logprobs=logprobs, ref_logprobs=ref_logprobs, masks=masks, step_num=step_num)

        return loss, stats

    def record_step_stats(self, data):
        masks = data['masks']
        kl = torch.sum(self.kl_loss(F.log_softmax(data['ref_logits'], dim=-1), F.softmax(data['logits'], dim=-1)), dim=-1)
        mean_kl = torch.mean(reduce_sum(kl, masks, axis=1))
        mean_entropy = torch.mean(reduce_sum(-data['logprobs'], masks, axis=1))
        stats = {
            'objective/kl': mean_kl.item(),
            'objective/entropy': mean_entropy.item(),
        }
        stats.update({
            'loss/total': data['total_loss'].item(),
            'loss/kl': data['kl_loss'].item(),
            'loss/lm': data['lm_loss'].item(),
            'loss/entropy': data['entropy'].item(),
        })

        return stats

    def print_samples(self, queries, responses, lm_loss, logprobs, ref_logprobs, masks, step_num):
        if step_num % self.params['logging']['log_interval'] != 0:
            return

        print(f"[step {step_num}] Printing samples examples ...")
        for i in range(min(3, len(queries))):
            sample_kl = torch.sum((logprobs[i] - ref_logprobs[i]) * masks[i]).item()
            print(f"\nSample {i+1}")
            print(f"{queries[i]} | {responses[i]}")
            print(f"  lm_loss = {lm_loss[i].item():+.2f}")
            print(f"  kl = {sample_kl:+.2f}")
            print(f"  total = {lm_loss[i].item() + self.params['env']['kl_coef'] * sample_kl:+.2f}")

    import heapq

    def save(self, step_num, eval_metric):
        if step_num % self.params['logging']['save_interval'] != 0:
            return

        eval_metric = float(eval_metric)

        # Remove the existing save directory and recreate it
        if os.path.exists(self.params["model_dir"]):
            shutil.rmtree(self.params["model_dir"])
        os.makedirs(self.params["model_dir"])

        # --- TOP MODELS ---
        if len(self.top_models) < self.top_models_limit:
            # If the list of top models is not full, add the current model
            heapq.heappush(self.top_models, (eval_metric, {
                'eval_metric': eval_metric,
                'policy_model': self.policy.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'step': step_num
            }))
        else:
            # If the list is full, compare the current model with the worst (lowest metric) in the top models
            worst_model = heapq.heappop(self.top_models)
            if eval_metric > worst_model[0]:
                # Replace the worst model with the current model
                heapq.heappush(self.top_models, (eval_metric, {
                    'eval_metric': eval_metric,
                    'policy_model': self.policy.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'step': step_num
                }))
            else:
                # Put the worst model back in the heap
                heapq.heappush(self.top_models, worst_model)

        for i, (metric, top_model) in enumerate(sorted(self.top_models, key=lambda x: x[0], reverse=True)):
            # Save the top models to separate files in the save_directory (overwrite them)
            model_filename = f'{self.params["model_dir"]}/top_model_{i}_step_{top_model["step"]}.pth'
            torch.save({
                'eval_metric': top_model['eval_metric'],
                'policy_model': top_model['policy_model'],
                'optimizer': top_model['optimizer'],
                'scheduler': top_model['scheduler']
            }, model_filename)
            print(f"Saved top model {i} with metric {top_model['eval_metric']} at step {top_model['step']}")

    def eval(self, step_num) -> Union[float, None]:
        if step_num % self.params['logging']['eval_interval'] != 0:
            return None
        print(f"[step {step_num}] Evaluating ...")

        prompts, responses = [], []
        references = []
        for i, batch in enumerate(tqdm(self.dev_dataloader, desc='(eval) Sampling from current policy')):
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

        print(f"Averaged Relevancy score for all dev_set samples = {rel_scores_mean_over_num_samples:+.2f}")
        print(f"Averaged Relevancy score for all dev_set sentences = {rel_scores_mean_over_num_sentences:+.2f}")
        print(f"(rel) Percentage of all subsentences in the dev_set predicted as 'no error' = {rel_correct_ratio:+.2f}")

        # FACTUALITY
        n_sentences = eval_output["n_sentences"]
        n_corrects_fact = eval_output["n_corrects_fact"]
        fact_rewards = eval_output["fact_rewards"]

        fact_scores_mean_over_num_samples = np.mean(fact_rewards) # averaged output score for all dev_set samples 
        fact_scores_mean_over_num_sentences = np.sum(fact_rewards) / np.sum(n_sentences) # averaged output score for all dev_set senteces
        fact_correct_ratio = np.sum(n_corrects_fact) / np.sum(n_sentences) # percentage of al sentences in the dev_set predicted as "no error"
        
        print(f"Averaged Factuality score for all dev_set samples = {fact_scores_mean_over_num_samples:+.2f}")
        print(f"Averaged Factuality score for all dev_set sentences = {fact_scores_mean_over_num_sentences:+.2f}")
        print(f"(fact) Percentage of all sentences in the dev_set predicted as 'no error' = {fact_correct_ratio:+.2f}")

        # COMPLETENESS
        comp_rewards = eval_output["comp_rewards"]

        comp_scores_mean_over_num_samples = np.mean(comp_rewards) # averaged output score for all dev_set samples 

        print(f"Averaged Completeness score for all dev_set samples = {comp_scores_mean_over_num_samples:+.2f}")

        # OTHERS
        generations_lens = eval_output["generations_lens"]
        rouge_scores = eval_output["rouge_scores"]

        avg_generations_lens = np.mean(generations_lens)
        avg_rouge_scores = np.mean(rouge_scores)

        print(f"Average generations lenght = {avg_generations_lens:+.2f}")
        print(f"Average RougeLSum = {avg_rouge_scores:+.2f}")

        if self.params['logging']['wandb_log']:
            wandb.log({f'Evaluation/rel_scores_mean_over_num_samples': rel_scores_mean_over_num_samples}, step=step_num)
            wandb.log({f'Evaluation/rel_scores_mean_over_num_sentences': rel_scores_mean_over_num_sentences}, step=step_num)
            wandb.log({f'Evaluation/rel_correct_ratio': rel_correct_ratio}, step=step_num)

            wandb.log({f'Evaluation/fact_scores_mean_over_num_samples': fact_scores_mean_over_num_samples}, step=step_num)
            wandb.log({f'Evaluation/fact_scores_mean_over_num_sentences': fact_scores_mean_over_num_sentences}, step=step_num)
            wandb.log({f'Evaluation/fact_correct_ratio': fact_correct_ratio}, step=step_num)

            wandb.log({f'Evaluation/comp_scores_mean_over_num_samples': comp_scores_mean_over_num_samples}, step=step_num)

            wandb.log({f'Evaluation/avg_len': avg_generations_lens}, step=step_num)
            wandb.log({f'Evaluation/avg_RougeLSum': avg_rouge_scores}, step=step_num)

        avg_score = (rel_scores_mean_over_num_samples + fact_scores_mean_over_num_samples + comp_scores_mean_over_num_samples) / 3
        return avg_score

def main():
    # set seed
    set_seed(
        seed=args['train']['seed'], 
        cuda_deterministic=args['train']['cuda_deterministic'])
    
    # set GPUs
    num_gpus = torch.cuda.device_count()
    print(f'Detected {num_gpus} GPUS')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    time = datetime.now()
    date_time = time.strftime("%m-%d-%Y_%H:%M:%S")

    # set wandb logging
    wandb_log = args['logging']['wandb_log']
    if wandb_log:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(
            entity=args['logging']['wandb_entity'],
            project=args['logging']['wandb_project'],
            name=f"{args['logging']['run_name']}_{date_time}"
        )
    # set saving directories
    args['save_dir'] = os.path.join(args['logging']['save_dir'], date_time)
    args['reward_dir'] = os.path.join(args['save_dir'], 'reward')
    args['model_dir'] = os.path.join(args['save_dir'], 'model')
    print(f"Writing to output directory: {args['save_dir']}")
    for dir in [args['save_dir'], args['reward_dir'], args['model_dir']]:
        ensure_dir(dir)

    # save the config file
    with open(os.path.join(args['save_dir'], 'args.json'), 'w') as f:
        json.dump(args, f, indent=2)
    
    # initialize tokenizer, policy to be finetuned, and reference policy
    print(f'Initializing models ...')
    model_name_or_path = args['model']['policy_model']['ckpt']
    nlf_cond = args['env']['nlf_cond']
    num_quantiles = args['env']['num_quantiles']
    num_attributes = args['env']['num_attributes']
    if nlf_cond:
        print("Conditioning on Natural Language Feedback tokens")
    else:
        print("Conditioning on Reward Quantile tokens (Quark-based)")
    print(f"Using {num_quantiles} quantiles for each of the {num_attributes} attributes.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=args['env']['max_input_len']
    )
    tokenizer.max_input_len = args['env']['max_input_len']
    tokenizer.max_generated_len = args['env']['max_generated_len']
    if nlf_cond:
        tokenizer.feedback_prefix = "feedback: "
        tokenizer.prompt_prefix = "input: "
        feedback_types = [
            # RELEVANCY
            [
                "Most relevant.",
                "Highly relevant.",
                "Moderately relevant.",
                "Slightly relevant.",
                "Least relevant."
            ],
            # FACTUALITY
            [
                "Most factual.",
                "Highly factual.",
                "Moderately factual.",
                "Slightly factual.",
                "Least factual."
            ],
            # COMPLETENESS
            [
                "Most complete.",
                "Highly complete.",
                "Moderately complete.",
                "Slightly complete.",
                "Least complete."
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

    ref_policy = T5Policy(
        model_ckpt=args['model']['ref_policy']['ckpt'],
        device=device,
        tokenizer=tokenizer,
    )

    policy = T5Policy(
        model_ckpt=model_name_or_path,
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

    data_pool = DataPool(
        feedback_types=feedback_types,
        num_quantiles=num_quantiles,
        num_attributes=num_attributes
    )

    # load datasets and dataloaders
    print(f'Loading data ...')
    prompt_collator = PromptCollator(tokenizer=tokenizer)

    train_dataset = PromptDataset(path=args['data']['train_data_path'])
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args['train']['sampling_batch_size_per_card'],
        shuffle=True,
        drop_last=True,
        collate_fn=prompt_collator
    )
    print(f"Train dataset loaded with {len(train_dataset)} samples | Train dataloader with {len(train_dataloader)} batches")

    dev_dataset = PromptDataset(path=args['data']['dev_data_path'])
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=args['train']['sampling_batch_size_per_card'],
        shuffle=False,
        drop_last=False,
        collate_fn=prompt_collator
    )
    print(f"Dev dataset loaded with {len(dev_dataset)} samples | Dev dataloader with {len(dev_dataloader)} batches")
    
    # prepare optimizer and schedulers
    optimizer = torch.optim.Adam(policy.model.parameters(), 
                                 lr=args['train']['lr'], eps = 1e-5)
    
    total_steps = ceil_div(args['train']['total_episodes'],
                           args['train']['training_batch_size_per_card'])
    
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=args['train']['n_warmup_steps'],
        num_training_steps=total_steps
    )

    # set up trainer
    trainer = ConditionOnFeedbackTrainer(
        params=args,
        policy=policy,
        ref_policy=ref_policy,
        score_model=reward_model,
        data_pool=data_pool,
        feedback_types=feedback_types,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        optimizer=optimizer,
        scheduler=scheduler
    )

    steps = list(range(total_steps + 1))
    steps = tqdm(steps)
    for step_num in steps:
        try:
            trainer.step(step_num)
        except Exception as e:
            print("There was an Exception while trying to perform trainer.step()!")
            print(e)
            torch.cuda.empty_cache()
            continue

if __name__ == "__main__":
    main()

    

