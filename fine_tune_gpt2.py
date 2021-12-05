import os
import time
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

"""Code Acknowledgement - sample code for working with Huggingface/GPT-2 was adapted from the following sources
https://towardsdatascience.com/text-generation-with-pretrained-gpt2-using-pytorch-563c7c90700
https://huggingface.co/docs/transformers/model_doc/gpt2
https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/gpt2_finetune_classification.ipynb
"""

def load_to_text(filepath):
    with open(filepath, "r", encoding='utf-8') as txt_file:
        text = txt_file.read()
        return text

def clean_text(text):
    text = text.replace('--', ' ')
    text = text.replace("\n", ' ')
    text = text.replace("?", '.')
    text = text.replace('!', '.')
    text = text.replace(',', '')
    text = text.replace(":", '')
    text = text.replace("-", '.')
    text = text.replace(";", '')
    text = text.lower()

    lines = text.split('.')
    max_len = 0
    for i in range(len(lines)):
        lines[i] = lines[i].lstrip()
        length = len(lines[i].split(' '))
        if length > max_len:
            max_len = length
    return lines

class PlatoDataset(Dataset):

    def __init__(self, lines, tokenizer, max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for line in lines:
            if len(line.split(' ')) < 5:
                continue
            else:
                encodings_dict = tokenizer('<|philosophy|>' + line + '<|endoftext|>',
                                           truncation=True, max_length=max_length, padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

if __name__ == '__main__':
    BATCH_SIZE = 2
    DATA_ROOT = "data/"

    epochs = 5
    learning_rate = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8

    # this produces sample output every 100 steps
    sample_every = 100
    lines = []
    for file in os.listdir(DATA_ROOT):
        if file.endswith(".txt"):
            print(file)
            text = load_to_text(DATA_ROOT + file)
            lines += clean_text(text)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|philosophy|>', eos_token="<|endoftext|>", pad_token='<|pad|>')

    dataset = PlatoDataset(lines, tokenizer, max_length=768)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    train_dataloader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        batch_size = BATCH_SIZE
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=BATCH_SIZE
    )

    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    # instantiate the model
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

    # Adjust size for added tokens
    model.resize_token_embeddings(len(tokenizer))

    # Run this model on the GPU.
    device = torch.device("cuda")
    model.cuda()

    seed_val = 64

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon
                      )

    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    # This changes the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    total_t0 = time.time()

    training_stats = []

    model = model.to(device)

    # Fine tune the model
    for epoch_i in range(0, epochs):

        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_masks,
                            token_type_ids=None
                            )

            loss = outputs[0]

            batch_loss = loss.item()
            total_train_loss += batch_loss

            loss.backward()

            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        # Run validation
        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                #                            token_type_ids=None,
                                attention_mask=b_masks,
                                labels=b_labels)

                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
            }
        )

    print("")
    print("Training complete!")

    # Display floats with two decimal places.
    pd.set_option('precision', 2)
    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])
    # Save the training plot
    plt.savefig("botcrates_gpt2_training.jpg")
    plt.show()

    output_dir = './model_save/'
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Lines to reload model
    # tokenizer = GPT2Tokenizer.from_pretrained("model_save/")
    # model = GPT2LMHeadModel.from_pretrained("model_save/")