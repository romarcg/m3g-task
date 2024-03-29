import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.data import DataLoader
from bitsandbytes.optim import Adam8bit
import math
from einops import rearrange
from tqdm import tqdm

from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image




class M3GDataset(Dataset):
    def __init__(self, split='train'):
        self.data = load_dataset(
            "json",
            data_files={"train": "data/llava_train_en_ds_ls.json", "val": "data/llava_train_en_ds_ls.json"},
        )[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": Image.open(sample["image"]), # Should be a PIL image
            "qa": [
                {
                    "question": sample["conversations"][0]["value"],
                    "answer": sample["conversations"][1]["value"],
                }
            ]
        }

# finetuning functions


def main():
    print (f"Loading datasets")
    datasets = {
    "train": M3GDataset("train"),
    "val": M3GDataset("val"),
    }

    # This Revision points to moondream2 (which overcomes our previoyus moonderam1 tests)
    DEVICE = "cuda"
    DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16 
    MD_REVISION = "2024-03-13"
    
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)
    moondream = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
        attn_implementation="flash_attention_2" if DEVICE == "cuda" else None,
        torch_dtype=DTYPE, device_map={"": DEVICE}
    )

    # follow moondream github to setup parameters
    EPOCHS = 1
    
    BATCH_SIZE = 8
    
    # Number of batches to process before updating the model. You can use this to simulate a higher batch
    # size than your GPU can handle. Set this to 1 to disable gradient accumulation.
    GRAD_ACCUM_STEPS = 1
    
    LR = 3e-5
    
    USE_WANDB = False

    # Main finetuning step

    def collate_fn(batch):
        images = [sample['image'] for sample in batch]
        images = torch.stack(moondream.vision_encoder.preprocess(images))
        images = rearrange(images,
                           "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                           p1=14, p2=14)
    
        labels_acc = []
        tokens_acc = []
    
        for sample in batch:
            toks = [tokenizer.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)
    
            for qa in sample['qa']:
                q_t = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))
    
                a_t = tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}",
                    add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)
    
            tokens_acc.append(toks)
            labels_acc.append(labs)
    
        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))
    
        attn_mask_acc = []
    
        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i
    
            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)
    
        return (
            images.to(dtype=DTYPE),
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )
    
    def compute_loss(batch):
        images, tokens, labels, attn_mask = batch
    
        images = images.to(DEVICE)
        tokens = tokens.to(DEVICE)
        labels = labels.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
    
        with torch.no_grad():
            img_embs = moondream.vision_encoder.encoder(images)
            img_embs = moondream.vision_encoder.projection(img_embs)
    
        tok_embs = moondream.text_model.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)
    
        outputs = moondream.text_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attn_mask,
        )
    
        return outputs.loss
    
    def lr_schedule(step, max_steps):
        x = step / max_steps
        if x < 0.1:
            return 0.1 * LR + 0.9 * LR * x / 0.1
        else:
            return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

    # The current version of moondream uses "<END>" to denote the end of a response. In the future this
    # will be replaced with a special token.
    ANSWER_EOS = "<END>"
    
    # Number of tokens used to represent each image.
    IMG_TOKENS = 729
    
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
        ),
    }

    print(f"Finetuning ...")
    moondream.text_model.train()
    moondream.text_model.transformer.gradient_checkpointing_enable()
    
    total_steps = EPOCHS * len(dataloaders["train"]) // GRAD_ACCUM_STEPS
    optimizer = Adam8bit(
        [
            {"params": moondream.text_model.parameters()},
        ],
        lr=LR * 0.1,
        betas=(0.9, 0.95),
        eps=1e-6
    )
    
    if USE_WANDB:
        import wandb
        wandb.init(
            project="moondream-ft",
            config={
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
                "LR": LR,
            }
        )
    
    i = 0
    for epoch in range(EPOCHS):
        for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            i += 1
    
            loss = compute_loss(batch)
            loss.backward()
    
            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
    
                lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
    
            if i % 100 == 0 and USE_WANDB:
                # Calculate validation loss
                val_loss = 0
                for val_batch in tqdm(dataloaders["val"], desc="Validation"):
                    with torch.no_grad():
                        val_loss += compute_loss(val_batch).item()
                val_loss /= len(dataloaders["val"])
    
            if USE_WANDB:
                wandb.log({
                    "loss/train": loss.item(),
                    "lr": optimizer.param_groups[0]['lr']
                } | ({"loss/val": val_loss} if i % 100 == 0 else {}))
    
    if USE_WANDB:
        wandb.finish()


    # save to checkpoints

    moondream.save_pretrained("checkpoints/moondream-m3g-ft")

    print (f"Done.")


if __name__ == "__main__":
    main()
