import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn as nn 
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
import evaluate as evaluate
from transformers import get_scheduler
from transformers import AutoModel, AutoModelForCausalLM
import argparse
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model

from data.mmlu.mmluwrapper import MMLUWrapper 


# Related to BERT
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings


class mmluDataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of BoolQ questions and answers
    """

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """

        example = self.data[index]
        question = example["question"]
        subject = example['subject']
        answer = example['mapped_answer']


        # input encoding for your model
        input_encoding = question

        # encode_plus will encode the input and return a dictionary of tensors
        encoded_review = self.tokenizer.encode_plus(
            input_encoding,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        
        return {
            'input_ids': encoded_review['input_ids'][0], 
            'attention_mask': encoded_review['attention_mask'][0],
            'answer': torch.tensor(answer, dtype=torch.long)  
        }


def evaluate_model(model, dataloader, device):
    """
    Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """
    # load metrics
    dev_accuracy = evaluate.load('accuracy')

    # turn model into evaluation mode
    model.eval()

    # iterate over the dataloader
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # forward pass
        output = model(input_ids=input_ids, attention_mask=attention_mask)

        predictions = output['logits']
        predictions = torch.argmax(predictions, dim=1)
        dev_accuracy.add_batch(predictions=predictions, references=batch['answer'])

    # compute and return metrics
    return dev_accuracy.compute()


def train(mymodel, num_epochs, train_dataloader, validation_dataloader, test_dataloder, device, lr, model_name):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :param string model_name: the name of the model
    :return None
    """

    # here, we use the AdamW optimizer. Use torch.optim.AdamW
    print(" >>>>>>>>  Initializing optimizer")
    
    optimizer = torch.optim.AdamW(mymodel.parameters(), lr=lr)
    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()
    
    epoch_list = []
    train_acc_list = []
    dev_acc_list = []

    for epoch in range(num_epochs):

        epoch_start_time = time.time()

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        train_accuracy = evaluate.load('accuracy')

        print(f"Epoch {epoch + 1} training:")

        for index, batch in tqdm(enumerate(train_dataloader)):

            """
            You need to make some changes here to make this function work.
            Specifically, you need to: 
            Extract the input_ids, attention_mask, and labels from the batch; then send them to the device. 
            Then, pass the input_ids and attention_mask to the model to get the logits.
            Then, compute the loss using the logits and the labels.
            Then, depending on model.type, you may want to use different optimizers
            Then, call loss.backward() to compute the gradients.
            Then, call lr_scheduler.step() to update the learning rate.
            Then, call optimizer.step()  to update the model parameters.
            Then, call optimizer.zero_grad() to reset the gradients for the next iteration.
            Then, compute the accuracy using the logits and the labels.
            """

            # TODO: implement the training loop
            # raise NotImplementedError("You need to implement this function")

            # get the input_ids, attention_mask, and labels from the batch and put them on the device
            # Hints: similar to the evaluate_model function
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            answer = batch['answer'].to(device)
            # forward pass
            # name the output as `output`
            # Hints: refer to the evaluate_model function on how to get the predictions (logits)
            # - It's slightly different from the implementation in train of base_classification.py
            output = mymodel(input_ids=input_ids, attention_mask=attention_mask)
            predictions = output['logits']
            # compute the loss using the loss function
            l = loss(predictions, answer)
            # loss backward
            l.backward()
            # your code ends here

            # update the model parameters depending on the model type
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            predictions = torch.argmax(predictions, dim=1)
            
            # update metrics
            train_accuracy.add_batch(predictions=predictions, references=batch['answer'])

        # print evaluation metrics
        print(f" ===> Epoch {epoch + 1}")
        train_acc = train_accuracy.compute()
        print(f" - Average training metrics: accuracy={train_acc}")
        train_acc_list.append(train_acc['accuracy'])

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(mymodel, validation_dataloader, device)
        print(f" - Average validation metrics: accuracy={val_accuracy}")
        dev_acc_list.append(val_accuracy['accuracy'])
        
        epoch_list.append(epoch)
        
        test_accuracy = evaluate_model(mymodel, test_dataloader, device)
        print(f" - Average test metrics: accuracy={test_accuracy}")

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} took {epoch_end_time - epoch_start_time} seconds")

def pre_process(model_name, batch_size, device, type='auto'):
    # download dataset
    print("Loading the dataset ...")
    mmlu_wrapper = MMLUWrapper()
    dataset = mmlu_wrapper.get_dataset()

    print("Loding the data into DS...")
    dataset_train = dataset['train']
    dataset_dev = dataset['dev']
    dataset_test = dataset['test']

    print("Loading the tokenizer...")
    mytokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    mytokenizer.pad_token = mytokenizer.eos_token # for generation

    max_len = 512

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(
        mmluDataset(dataset_train_subset, mytokenizer, max_len),
        batch_size=batch_size,
    )
    validation_dataloader = DataLoader(
        mmluDataset(dataset_dev_subset, mytokenizer, max_len),
        batch_size=batch_size
    )
    test_dataloader = DataLoader(
        mmluDataset(dataset_test_subset, mytokenizer, max_len),
        batch_size=batch_size
    )

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)    
    # TODO: Task type currently set as QUESTION_ANS, CHANGE RANK
    peft_config = LoraConfig(task_type=TaskType.QUESTION_ANS, inference_mode=False, r=6, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "key", "value"])
    pretrained_model = get_peft_model(pretrained_model, peft_config)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_subset", action='store_true')
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    assert type(args.small_subset) == bool, "small_subset must be a boolean"
    #load the data and models
    pretrained_model, train_dataloader, validation_dataloader, test_dataloader = pre_process(args.model,
                                                                                             args.batch_size,
                                                                                             args.device,
                                                                                             args.small_subset,
                                                                                            )
    print(" >>>>>>>>  Starting training ... ")
    train(pretrained_model, args.num_epochs, train_dataloader, validation_dataloader, test_dataloader, args.device, args.lr, args.model)

    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")
