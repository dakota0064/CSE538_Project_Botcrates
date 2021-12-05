
import random
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel

if __name__ == '__main__':

    device = torch.device("cuda")

    tokenizer = GPT2Tokenizer.from_pretrained("model_save/")
    model = GPT2LMHeadModel.from_pretrained("model_save/")
    model.cuda()

    prompt = "<|philosophy|>"

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    sample_outputs = model.generate(
        generated,
        bos_token_id=random.randint(1,30000),
        do_sample=True,
        top_k=100,
        max_length=300,
        top_p=0.98,
        num_return_sequences=150
    )

    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))