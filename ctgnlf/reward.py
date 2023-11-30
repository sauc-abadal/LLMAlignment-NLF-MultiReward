from .my_longformer import LongformerForSequenceClassification, LongformerForTokenClassification
from .reward_utils import split_text_to_subsentences, split_text_to_sentences
from .evaluators import get_rouge_scores
from .utils import batchify

from typing import Union, List
import os
from transformers import AutoTokenizer
import torch
import spacy
from tqdm import tqdm

class MyBaseRewardModel:
    def __init__(self,
                 policy_tokenizer: AutoTokenizer, 
                 reward_model_name_or_path: Union[str, os.PathLike], 
                 device: torch.device,
                 positive_reward: float,
                 negative_reward: float,
                 split_function):
        
        self.device = device

        # prepare policy tokenizer
        self.policy_tokenizer = policy_tokenizer

        # prepare reward tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name_or_path)
        
        # prepare factual reward model
        self.reward_model = LongformerForTokenClassification.from_pretrained(reward_model_name_or_path).to(device)
        
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()
        
        # prepare spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.sep = "</s>"
        
        # the id corresponds to the sep token
        self.sep_id = self.reward_tokenizer.convert_tokens_to_ids(self.sep)

        # rewards
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

        # split function 
        self.split_function = split_function
    
    def find_sep_position(self, input_ids):
        return torch.nonzero(input_ids == self.sep_id, as_tuple=False).squeeze().tolist()

    def process_one_generation(self, long_text, policy_text_len):
        
        sentence_end_char_idxs= self.split_function(long_text, self.nlp)
            
        sentences = [long_text[sentence_end_char_idxs[i]:sentence_end_char_idxs[i+1]] for i in range(len(sentence_end_char_idxs)-1)]
        
        # Initialize an empty list to store the end token indices of each sentence
        sentence_end_indices = []

        for sent_idx in range(len(sentences)):
            tokens = self.policy_tokenizer.tokenize(long_text[:sentence_end_char_idxs[sent_idx+1]])
            token_count = len(tokens)
            sentence_end_indices.append(token_count - 1)
        
        reward_sentences = [f"{self.sep} {sent}" for sent in sentences]

        reward_input = ' '.join(reward_sentences)
        
        # in case overlength    
        sentence_end_indices = [min(item, policy_text_len-1) for item in sentence_end_indices]

        return reward_input, sentence_end_indices
    
    def get_full_reward(self, 
                        prompt_texts: List[str], 
                        generated_texts: List[str]):
        raise NotImplementedError("Subclasses must implement this method.")

    def get_reward(self, 
                   prompt_texts: List[str],
                   generated_texts: List[str],
                   ):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_reward_batch(self,
                         prompt_texts: List[str],
                         generated_texts: List[str],
                         batch_size: int
                         ):
        pbar = tqdm(total=len(prompt_texts), dynamic_ncols=True)
        pbar.set_description("Computing rewards in batches...")
        rewards = []
        for batch in zip(batchify(prompt_texts, batch_size=batch_size), batchify(generated_texts, batch_size=batch_size)):
            prompt_texts_batch, generated_texts_batch = batch
            rewards.extend(self.get_reward(prompt_texts_batch, generated_texts_batch)["rewards"])
            pbar.update(len(prompt_texts_batch))
        return rewards

class MyFactualityRewardModel(MyBaseRewardModel):
    def __init__(self, 
                 policy_tokenizer: AutoTokenizer, 
                 reward_model_name_or_path: Union[str, os.PathLike], 
                 device: torch.device,
                 positive_reward: float,
                 negative_reward: float,
                 split_function=split_text_to_sentences):
        
        super().__init__(
            policy_tokenizer, 
            reward_model_name_or_path, 
            device,
            positive_reward,
            negative_reward,
            split_function
        )

    def get_full_reward(self, 
                   prompt_texts: List[str],
                   generated_texts: List[str],
                   ):
        
        batch_reward_inputs = []
        batch_sentence_end_indices = []
        
        # get the length of generated outputs
        generated_attention_mask = self.policy_tokenizer(generated_texts, padding=True, truncation=True, return_tensors="pt").attention_mask
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()

        for batch_idx, (prompt_text, gen_text) in enumerate(zip(prompt_texts, generated_texts)):
            reward_input, sentence_end_indices = self.process_one_generation(gen_text, policy_inputs_lens[batch_idx])

            # input for the factual reward model
            f_reward_input = f"{prompt_text} answer: {reward_input}"
            batch_reward_inputs.append(f_reward_input)
            
            # the indices of sentence ends for the policy model output
            batch_sentence_end_indices.append(sentence_end_indices)
        
        # get the reward
        with torch.no_grad():
            
            # to align with the token classification model
            inputs =self.reward_tokenizer([s.split() for s in batch_reward_inputs], 
                                            truncation=True, padding=True, 
                                            is_split_into_words=True,
                                            return_tensors="pt")
            inputs = inputs.to(self.reward_model.device)
            
            # factual reward model
            batch_f_pred = self.reward_model(**inputs)
            
        factuality_rewards = []
        n_corrects = []
        
        for text_idx, generated_text in enumerate(generated_texts):
            
            # extract the rewards from factual reward model output
            this_f_pred = batch_f_pred.logits[text_idx].detach().cpu()
            
            generated_text_input_ids = self.reward_tokenizer(
                batch_reward_inputs[text_idx].split(), 
                return_tensors="pt", 
                is_split_into_words=True,
                truncation=True).input_ids[0]
            
            # get the indices of </s>
            sep_indices = self.find_sep_position(generated_text_input_ids)
            sentence_f_reward_probs = this_f_pred[sep_indices]
            
            # align back to the original sentence
            policy_sentence_end_indices = batch_sentence_end_indices[text_idx]
            policy_inputs_len = policy_inputs_lens[text_idx]
            
            this_factuality_reward = [0]*policy_inputs_len
            
            this_n_correct = 0
            
            for i, end_idx in enumerate(policy_sentence_end_indices):
                
                # 0 is has error, 1 is no error
                f_error_type = torch.argmax(sentence_f_reward_probs[i][[0,2]]).item()
                factuality_reward = self.positive_reward if f_error_type == 1 else self.negative_reward
                
                # aggregate the rewards
                this_factuality_reward[end_idx] = factuality_reward
                
                if f_error_type == 1:
                    this_n_correct += 1
                    
            n_corrects.append(this_n_correct)
                
            factuality_rewards.append(this_factuality_reward)
            
        return {"factuality_rewards": factuality_rewards,
                "n_sentences": [len(item) for item in batch_sentence_end_indices],
                "n_corrects": n_corrects}
    
    def get_reward(self, 
                   prompt_texts: List[str],
                   generated_texts: List[str],
                   ):
        
        rewards_output = self.get_full_reward(prompt_texts, generated_texts)["factuality_rewards"]
        pooled_reward = [sum(x) for x in rewards_output]
        return {'rewards': pooled_reward}

class MyRelevancyRewardModel(MyBaseRewardModel):
    def __init__(self, 
                 policy_tokenizer: AutoTokenizer, 
                 reward_model_name_or_path: Union[str, os.PathLike], 
                 device: torch.device,
                 positive_reward: float,
                 negative_reward: float,
                 split_function=split_text_to_subsentences):
        
        super().__init__(
            policy_tokenizer, 
            reward_model_name_or_path, 
            device,
            positive_reward,
            negative_reward,
            split_function
        )

    def get_full_reward(self, 
                   prompt_texts: List[str],
                   generated_texts: List[str],
                   ):
        
        batch_reward_inputs = []
        batch_sentence_end_indices = []
        
        # get the length of generated outputs
        generated_attention_mask = self.policy_tokenizer(generated_texts, padding=True, truncation=True, return_tensors="pt").attention_mask
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()

        for batch_idx, (prompt_text, gen_text) in enumerate(zip(prompt_texts, generated_texts)):
            reward_input, sentence_end_indices = self.process_one_generation(gen_text, policy_inputs_lens[batch_idx])

            # input for the (non-factual) reward model
            split_prompt = prompt_text.split("context: ", 1)
            if len(split_prompt) > 1: # "context: " found
                question = split_prompt[0].strip() # get part before context, i.e. question
            else:
                question = prompt_text.strip() # pass whole prompt if not found
            nf_reward_input = f"{question} answer: {reward_input}"
            batch_reward_inputs.append(nf_reward_input)
            
            # the indices of sentence ends for the policy model output
            batch_sentence_end_indices.append(sentence_end_indices)
        
        # get the reward
        with torch.no_grad():
            
            # to align with the token classification model
            inputs =self.reward_tokenizer([s.split() for s in batch_reward_inputs], 
                                            truncation=True, padding=True, 
                                            is_split_into_words=True,
                                            return_tensors="pt")
            inputs = inputs.to(self.reward_model.device)
            
            # factual reward model
            batch_nf_pred = self.reward_model(**inputs)
            
        relevancy_rewards = []
        n_corrects = []
        
        for text_idx, generated_text in enumerate(generated_texts):
            
            # extract the rewards from factual reward model output
            this_nf_pred = batch_nf_pred.logits[text_idx].detach().cpu()
            
            generated_text_input_ids = self.reward_tokenizer(
                batch_reward_inputs[text_idx].split(), 
                return_tensors="pt", 
                is_split_into_words=True,
                truncation=True).input_ids[0]
            
            # get the indices of </s>
            sep_indices = self.find_sep_position(generated_text_input_ids)
            sentence_nf_reward_probs = this_nf_pred[sep_indices]
            
            # align back to the original sentence
            policy_sentence_end_indices = batch_sentence_end_indices[text_idx]
            policy_inputs_len = policy_inputs_lens[text_idx]
            
            this_relevancy_reward = [0]*policy_inputs_len
            
            this_n_correct = 0
            
            for i, end_idx in enumerate(policy_sentence_end_indices):
                
                # 0 is has error, 1 is no error
                nf_error_type = torch.argmax(sentence_nf_reward_probs[i][[1,2]]).item()
                relevancy_reward = self.positive_reward if nf_error_type == 1 else self.negative_reward
                
                # aggregate the rewards
                this_relevancy_reward[end_idx] = relevancy_reward
                
                if nf_error_type == 1:
                    this_n_correct += 1
                    
            n_corrects.append(this_n_correct)
                
            relevancy_rewards.append(this_relevancy_reward)
            
        return {"relevancy_rewards": relevancy_rewards,
                "n_sub_sentences": [len(item) for item in batch_sentence_end_indices],
                "n_corrects": n_corrects}
    
    def get_reward(self, 
                   prompt_texts: List[str],
                   generated_texts: List[str],
                   ):
        
        rewards_output = self.get_full_reward(prompt_texts, generated_texts)["relevancy_rewards"]
        pooled_reward = [sum(x) for x in rewards_output]
        return {'rewards': pooled_reward}

class MyCompletenessRewardModel:
    def __init__(self,
                 policy_tokenizer: AutoTokenizer, 
                 reward_model_name_or_path: Union[str, os.PathLike], 
                 device: torch.device,
                 mean: float = 0.0,
                 std: float = 1.0,
                 bias: float = 0.0,
                 scale: float = 1.0,
                 ):

        self.device = device

        # prepare policy tokenizer
        self.policy_tokenizer = policy_tokenizer
        
        # prepare reward tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name_or_path)
        
        self.model = LongformerForSequenceClassification.from_pretrained(reward_model_name_or_path).to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # use mean and std to normalize the reward
        # use bias and scale to rescale the reward
        self.mean = mean
        self.std = std
        self.bias = bias
        self.scale = scale
        
    def get_reward(self,
                   prompt_texts: List[str],
                   generated_texts: List[str]):
        
        batch_reward_inputs = []

        for batch_idx, (prompt_text, gen_text) in enumerate(zip(prompt_texts, generated_texts)):
            reward_input = f"{prompt_text} answer: {gen_text}"
            batch_reward_inputs.append(reward_input)
        
        # get the reward
        with torch.no_grad():
            # to align with the token classification model
            inputs =self.reward_tokenizer(batch_reward_inputs, 
                                          truncation=True, padding=True, 
                                          return_tensors="pt")
            inputs = inputs.to(self.model.device)
            outputs = self.model(**inputs)
            sequence_level_reward = outputs['logits'].squeeze(-1).tolist() 
        
        return {'rewards': [((r-self.mean)/self.std)*self.scale + self.bias for r in sequence_level_reward]}
    
    def get_reward_batch(self,
                         prompt_texts: List[str],
                         generated_texts: List[str],
                         batch_size: int
                         ):
        pbar = tqdm(total=len(prompt_texts), dynamic_ncols=True)
        pbar.set_description("Computing rewards in batches...")
        rewards = []
        for batch in zip(batchify(prompt_texts, batch_size=batch_size), batchify(generated_texts, batch_size=batch_size)):
            prompt_texts_batch, generated_texts_batch = batch
            rewards.extend(self.get_reward(prompt_texts_batch, generated_texts_batch)["rewards"])
            pbar.update(len(prompt_texts_batch))
        return rewards

class MyFineGrainedRewardModel:
    
    def __init__(self,
                 relevancy_reward_model: MyRelevancyRewardModel,
                 factuality_reward_model: MyFactualityRewardModel,
                 completeness_reward_model: MyCompletenessRewardModel
                 ):
        
        self.relevancy_rm = relevancy_reward_model
        self.factuality_rm = factuality_reward_model
        self.completeness_rm = completeness_reward_model

    def get_reward(self,
                   prompt_texts: List[str],
                   generated_texts: List[str]
                   ) -> List[List[float]]:
        
        rel_rewards = self.relevancy_rm.get_reward(prompt_texts, generated_texts)["rewards"]
        fact_rewards = self.factuality_rm.get_reward(prompt_texts, generated_texts)["rewards"]
        comp_rewards = self.completeness_rm.get_reward(prompt_texts, generated_texts)["rewards"]

        rewards = [rel_rewards, fact_rewards, comp_rewards]

        return rewards
    
    def get_reward_batch(self,
                         prompt_texts: List[str],
                         generated_texts: List[str],
                         batch_size: int) -> List[List[float]]:
    
        pbar = tqdm(total=len(prompt_texts), dynamic_ncols=True)
        pbar.set_description("Computing finegrained rewards in batches...")
        rel_rewards, fact_rewards, comp_rewards = [], [], []
        for batch in zip(batchify(prompt_texts, batch_size=batch_size), batchify(generated_texts, batch_size=batch_size)):
            prompt_texts_batch, generated_texts_batch = batch
            rel_rewards.extend(self.relevancy_rm.get_reward(prompt_texts_batch, generated_texts_batch)["rewards"])
            fact_rewards.extend(self.factuality_rm.get_reward(prompt_texts_batch, generated_texts_batch)["rewards"])
            comp_rewards.extend(self.completeness_rm.get_reward(prompt_texts_batch, generated_texts_batch)["rewards"])
            pbar.update(len(prompt_texts_batch))

        rewards = [rel_rewards, fact_rewards, comp_rewards]   
        return rewards
    
    def get_metrics(self,
                    prompt_texts: List[str],
                    generated_texts: List[str],
                    batch_size: int,
                    references: List[List[str]]):
        
        pbar = tqdm(total=len(prompt_texts), dynamic_ncols=True)
        pbar.set_description("Computing full rewards in batches...")
        n_sub_sentences, n_sentences, n_corrects_rel, n_corrects_fact, rel_rewards, fact_rewards, comp_rewards = [], [], [], [], [], [], []
        for batch in zip(batchify(prompt_texts, batch_size=batch_size), batchify(generated_texts, batch_size=batch_size)):
            prompt_texts_batch, generated_texts_batch = batch

            # RELEVANCY
            full_rel_reward_batch = self.relevancy_rm.get_full_reward(prompt_texts_batch, generated_texts_batch)
            n_sub_sentences_batch = full_rel_reward_batch ['n_sub_sentences']
            n_corrects_rel_batch = full_rel_reward_batch['n_corrects']
            rel_rewards_batch = [sum(x) for x in full_rel_reward_batch['relevancy_rewards']]

            n_sub_sentences.extend(n_sub_sentences_batch)
            n_corrects_rel.extend(n_corrects_rel_batch)
            rel_rewards.extend(rel_rewards_batch)

            # FACTUALITY
            full_fact_reward_batch = self.factuality_rm.get_full_reward(prompt_texts_batch, generated_texts_batch)
            n_sentences_batch = full_fact_reward_batch ['n_sentences']
            n_corrects_fact_batch = full_fact_reward_batch['n_corrects']
            fact_rewards_batch = [sum(x) for x in full_fact_reward_batch['factuality_rewards']]

            n_sentences.extend(n_sentences_batch)
            n_corrects_fact.extend(n_corrects_fact_batch)
            fact_rewards.extend(fact_rewards_batch)

            # COMPLETENESS
            comp_rewards_batch = self.completeness_rm.get_reward(prompt_texts_batch, generated_texts_batch)['rewards']

            comp_rewards.extend(comp_rewards_batch)


            pbar.update(len(prompt_texts_batch))
        
        generations_lens = [len(self.policy_tokenizer.encode(text)) for text in generated_texts]
        rouge_scores = get_rouge_scores(generated_texts, references)

        output = {
            "n_sub_sentences": n_sub_sentences,
            "n_corrects_rel": n_corrects_rel,
            "rel_rewards": rel_rewards,
            "n_sentences": n_sentences,
            "n_corrects_fact": n_corrects_fact,
            "fact_rewards": fact_rewards,
            "comp_rewards": comp_rewards,
            "generations_lens": generations_lens,
            "rouge_scores": rouge_scores
        }
        return output