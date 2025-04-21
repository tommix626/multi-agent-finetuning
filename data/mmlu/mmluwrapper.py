from datasets import load_dataset, DatasetDict

# trainset is for part1 (clustering and fine-tuning)
# devset is for training of part2 (prompt-tuning, mixer)
# testset is for evaluation after part2 (overall evaluation)

class MMLUWrapper:
    def __init__(self, seed=42, test_size=0.2):
        self.seed = seed
        self.test_size = test_size
        self.dataset = self._load_and_process()

    def _load_and_process(self):
        raw = load_dataset("cais/mmlu", "all")

        train_set  = self._map_choices_to_answer(raw["test"])
        aux_data = self._map_choices_to_answer(raw["auxiliary_train"])

        aux_split = aux_data.train_test_split(
            test_size=self.test_size, seed=self.seed
        )
        dev_set = aux_split["train"]
        test_set = aux_split["test"]

        return DatasetDict({
            "train": train_set,
            "dev": dev_set,
            "test": test_set
        })

    def _map_choices_to_answer(self, dataset):
        def _mapper(example):
            answer_index = example["answer"]
            answer_str = example["choices"][answer_index]
            return {"mapped_answer": answer_str}
        return dataset.map(_mapper)

    def get_dataset(self):
        return self.dataset

    def get_splits(self):
        return self.dataset["train"], self.dataset["dev"], self.dataset["test"]
    
    def get_train(self):
        return self.dataset["train"]

    def get_dev(self):
        return self.dataset["dev"]

    def get_test(self):
        return self.dataset["test"]
    

