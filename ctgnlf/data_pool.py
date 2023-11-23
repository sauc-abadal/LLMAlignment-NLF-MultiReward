from typing import List
from copy import deepcopy
from collections import defaultdict

class DataPool:
    def __init__(self, feedback_types, num_quantiles, num_attributes):
        """
        Initialize a data pool for organizing and managing data into quantiles.

        Args:
            feedback_types (List[List[str]]): A list of possible feedback/reward tokens associated with each quantile,
            for each attribute we care of, e.g., Relevancy, Factuality, and Completeness.
            num_quantiles (int): The number of quantiles to divide the data pool into.

        Note:
            The `feedback_types` list should contain feedback associated with each quantile (len(feedback_types) == num_quantiles).

            Quark-like:
                num_quantiles = 5
                num_attributes = 1
                feedback_types = [
                    [_TREE_TOKEN_00000, _TREE_TOKEN_00001, _TREE_TOKEN_00002, _TREE_TOKEN_00003, _TREE_TOKEN_00004],
                ]
            
            NLF (Factuality):
                num_quantiles = 5
                num_attributes = 1
                feedback_types = [
                    [Very factual, Majorly factual, Moderately factual, Slightly factual, Completely factual],
                ]
        """
        self.feedback_types = feedback_types
        self.num_quantiles = num_quantiles
        self.num_attributes = num_attributes

        self.score_pool = defaultdict(list)
        for i in range(self.num_attributes):
            self.score_pool[f"attr_{str(i)}"]
        self.prompt_pool, self.response_pool, self.feedback_pool = [], [], []

    def add(self, prompts: List[str], responses: List[str], scores: List[List[float]]):
        """
        Add data to the data pool and organize it into quantiles.

        Args:
            prompts (List[str]): A list of input prompts.
            responses (List[str]): A list of response sequences.
            scores (List[List[float]]): A list of reward scores (i.e., Relevancy, Factuality, and Completeness) 
            corresponding to the responses for every attribute.

        Note:
            - Data is sorted by reward scores, from highest to lowest reward, and feedback/reward tokens are assigned to samples based on quantile ranking.
            - Quantile 0 is associated with highest reward (e.g., highest relevancy), and Quantile 4 is associated with lowest reward (e.g., lowest relevancy)!
        """
        self.prompt_pool.extend(prompts)
        self.response_pool.extend(responses)
        for i, scores_list in enumerate(scores):
            self.score_pool[f"attr_{str(i)}"].extend(scores_list)
        self.feedback_pool = ["" for _ in range(len(self.prompt_pool))]

        for attr_type in range(self.num_attributes):
            data = zip(self.prompt_pool, self.response_pool, self.feedback_pool, self.score_pool[f"attr_{str(attr_type)}"])
            data = [x for x in data if x[-1] is not None]

            # get the sorting indices corresponding to sorting the data according to current attr_type
            sorted_indices = [i for i, x in sorted(enumerate(data), key=lambda x: x[1][-1], reverse=True)]

            # update pool of prompts, responses, feedback, and scores for current attr_type, according to current attr_type
            sorted_data = [data[i] for i in sorted_indices]
            self.prompt_pool, self.response_pool, self.feedback_pool, self.score_pool[f"attr_{str(attr_type)}"] = [list(x) for x in list(zip(*sorted_data))]
            # update pool of scores for all other attr_types, according to current_attr_type
            for j in range(self.num_attributes):
              if j != attr_type:
                self.score_pool[f"attr_{str(j)}"] = [self.score_pool[f"attr_{str(j)}"][i] for i in sorted_indices]
            
            # divide data pool into quantiles of roughly equal size (last quantile will be larger if the length of the data is not 
            # divisible by the desired number of quantiles), and obtain the associated quantile index to each sample in the data pool
            quantile_idx = [[i] * (len(sorted_data) // self.num_quantiles) for i in range(self.num_quantiles)]
            quantile_idx = [y for x in quantile_idx for y in x] # unfold list of lists into a single list
            quantile_idx = quantile_idx + [self.num_quantiles - 1] * (len(sorted_data) - len(quantile_idx)) # append indices for the last quantile
            # e.g., quantile_idx will be [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4] if currently the data pool has length 14 and we want to use 5 quantiles (the last four '4's are added as 14 % 5 != 0)
            
            self.feedback_pool = [(self.feedback_pool[i] + " " + self.feedback_types[attr_type][idx]).strip() for i, idx in enumerate(quantile_idx)] 

    def get_data(self):
        """
        Get the data from the data pool.

        Returns:
            Tuple[List[str], List[str], List[str]: A tuple containing the input prompts, response sequences,
            and feedback associated with quantiles.

        """
        return deepcopy(self.prompt_pool), deepcopy(self.response_pool), deepcopy(self.feedback_pool)

