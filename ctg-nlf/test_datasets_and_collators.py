from datasets_and_collators import PromptDataset, PromptCollator, SequenceWithFeedbackDataset, SequenceWithFeedbackCollator 
from data_pool import DataPool
from policy import T5Policy
import torch
from torch.utils.data import DataLoader
from reward import MyFactualityRewardModel

root_data_path = "/cluster/project/sachan/sauc/MultiTask-CTG-NLF/tasks/qa_feedback/data"
train_dataset = PromptDataset(path=root_data_path+"/train.json")
print(len(train_dataset))
dev_dataset = PromptDataset(path=root_data_path+"/dev.json")
print(len(dev_dataset))
test_dataset = PromptDataset(path=root_data_path+"/test.json")
print(len(test_dataset))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_name_or_path = "/cluster/project/sachan/sauc/MultiTask-CTG-NLF/tasks/qa_feedback/model_outputs/t5-large-1k-train"
nlf_cond = False
policy = T5Policy(
    model_ckpt=model_name_or_path, 
    device=device,
    temperature=1.0,
    nlf_cond=nlf_cond
)
tokenizer = policy.tokenizer

reward_model_name_or_path = "/cluster/project/sachan/sauc/MultiTask-CTG-NLF/tasks/qa_feedback/model_outputs/fact_rm"
reward_model = MyFactualityRewardModel(tokenizer=tokenizer, reward_model_name_or_path=reward_model_name_or_path, device=device)

num_quantiles = 5
num_attributes = 1
feedback_types = [
    ["Factuality_Q0", "Factuality_Q1", "Factuality_Q2", "Factuality_Q3", "Factuality_Q4"],
]
data_pool = DataPool(feedback_types=feedback_types, num_quantiles=num_quantiles, num_attributes=num_attributes)

seq_collator = SequenceWithFeedbackCollator(tokenizer=policy.tokenizer)
prompt_collator = PromptCollator(tokenizer=tokenizer)
batch_size = 20
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=prompt_collator)
prompts, responses = [], []
for idx, batch in enumerate(dev_dataloader):
    if idx == 1:
        break
    print(batch)
    prompts_input_ids, prompts_attention_mask = batch
    rollouts = policy.sample(prompts_input_ids=prompts_input_ids,
                             prompts_attention_mask=prompts_attention_mask,
                             do_sample=True, top_p=0.9)
    prompt, response = rollouts["prompts_text"], rollouts["generated_text"]
    prompts.extend(prompt)
    responses.extend(response)

import pdb
pdb.set_trace()
scores = reward_model.get_reward_batch(prompts, responses, batch_size=4)

data_pool.add(prompts=prompts, responses=responses, scores=[scores])

sample_dataset = SequenceWithFeedbackDataset(data_pool=data_pool)
sample_dataloader = DataLoader(sample_dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=seq_collator)
sampler = iter(sample_dataloader)

batch = next(sampler)
print(tokenizer.batch_decode(batch[0]))

print(batch)