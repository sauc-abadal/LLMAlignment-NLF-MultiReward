# LLM Alignment with NLF (Multi Reward)

This is the repo for our research on LLM Alignment with Natural Language Feedback. In this repo work on a preliminarly step to assess the viability of our approach and use synthetically defined language tags instead of full-blown natural language feedback. The language tags are obtained by leveraging a Reward Model to quantize reward scores assigned to LLM-generared outputs and further mapping the quantiles into tags. Notice that this poses a limitation in our approach, as we are restricted by the performance of the Reward Model and the expressibility of the language tags. We tackle the Long-Form QA task as in the [FGRLHF paper](https://arxiv.org/pdf/2306.01693.pdf), as a **multi-task reward** setting consisting of optimizng towards more relevant, factual, and complete LLM-generated answers.

On the LFQA task, during training we employed 3 quantiles which we further mapped into the following language tags:

```
tags = [
            # RELEVANCY
            [
                "relevant",
                "some irrelevancies",
                "irrelevant",
            ],
            # FACTUALITY
            [
                "factual",
                "some inaccuracies",
                "very inaccurate",
            ],
            # COMPLETENESS
            [
                "complete",
                "some details missing",
                "incomplete",
            ],
        ]
```
The language feedback sentence prepended during sampling and training is composed by leveraging the composibility property of language, i.e., adding ", and " in between tags, e.g., "relevant, and some inaccuracies, and some details missing".

Then, at inference time, we conditioned on the feedback associated with the highest-reward quantiles, i.e., "relevant, and factual, and complete".

## Set Up
```bash
# create a conda environment with python 3.9
conda create --name py39 python=3.9
conda activate py39 

# git clone and install packages
git clone https://github.com/sauc-abadal/LLMAlignment-NLF-MultiReward.git
cd MultiTask-CTG-NLF
pip install -e .
python -m spacy download en_core_web_sm
```

## SFT Training
The policy model is initialized with supervised finetuning on 1K training examples, by running the following command. The trained model is saved under `./tasks/qa_feedback/model_outputs/t5-large-1k-train`.

```bash
bash tasks/qa_feedback/training/train_sft.sh
```

## Reward Modeling
The fine-grained Reward Models employed, R1, R2 and R3 that correspond to (1) irrelevance, repetition, and incoherence error and (2) incorrect or unverifiable facts and (3) information completeness, are taken from the [FGRLHF repo](https://github.com/allenai/FineGrainedRLHF). You can train them by running the following commands:

```bash
# prepare RM training data, all data saved under ./tasks/qa_feedback/data
bash tasks/qa_feedback/reward_modeling/create_rm_train_files.sh

# train R1, saved under ./tasks/qa_feedback/model_outputs/rel_rm
bash tasks/qa_feedback/reward_modeling/train_rel_rm.sh

# train R2, saved under ./tasks/qa_feedback/model_outputs/fact_rm
bash tasks/qa_feedback/reward_modeling/train_fact_rm.sh

# train R3, saved under ./tasks/qa_feedback/model_outputs/comp_rm
bash tasks/qa_feedback/reward_modeling/train_comp_rm.sh
```

## Training
We provide training scripts to run both our NLF approach and our implementation of an extension of [Quark](https://github.com/GXimingLu/Quark) to the multidimensional setting, which, as opposed to our approach, uses newly added reward-based quantile tokens instead of language tags. We experiment with different sampling strategies motivated by the hypothesis that our approach might benefit from the composibility property of language and leverage learning individual representations of the different attributes, and compose them together at inference time. {approach} can be either "ctgnlg" or "quark". {sampling_strategy} can be one of "bestOf3", "bestOf2", "bestOf2_allAttributes", "bestOf1", and "bestOf1_allAttributes". For "allAttributes" strategies, you should first move to the "sampling_v2" branch. We found that "bestOf1_allAttributes" worked best for our approach while "bestO3" worked best for Quark.

You can find the hyperparameters in  `tasks/qa_feedback/training/multitask_{approach}_{sampling_strategy}_3quantiles_newTags_config.yml` and change them accordingly. For example, you may want to change `wandb_entity` as your own wandb username. 

You also want to change the `mean` and `std` values for sequence-level reward models (preference and info completeness) in each yml file, which stand for the mean and average reward scores on the training data from the reward model. You can find the values from the `mean_std.txt` file under `./tasks/qa_feedback/model_outputs/comp_rm` or `./tasks/qa_feedback/model_outputs/baseline_rm`. The current values are from the already trained reward models.

For qa-feedback:
```bash
bash tasks/qa_feedback/training/train_multitask_{approach}_{sampling_strategy}_3quantiles_newTags.sh --config tasks/qa_feedback/training/multitask_{approach}_{sampling_strategy}_3quantiles_newTags_config.yml
```

## Evaluation
We provide evaluation scripts in `tasks/qa_feedback/evaluation`.

## Trained Models

You can find the initial trained SFT 1K policy, and the reward models in the [FGRLHF repo](https://github.com/allenai/FineGrainedRLHF). Place the unzipped folders inside the `./task/qa_feedback/model_outputs` folder and run the scripts above that depend on these models accordingly. 

We provide our trained models in [here](link).
