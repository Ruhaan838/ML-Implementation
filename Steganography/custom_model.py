
import pandas as pd
import json
import os
import math
from PIL import Image
import numpy as np
from pathlib import Path

import torch
import torchmetrics
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from transformers import BertModel, ViTModel

from dataclasses import dataclass
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu


@dataclass
class Paths:
    root = "/kaggle/input/coco-2017-dataset/coco2017"
    train_cap = root + "/annotations/captions_train2017.json"
    train_img = root + "/train2017"

    val_cap = root + "/annotations/captions_val2017.json"
    val_img = root + "/val2017"

@dataclass
class Config:
    d_model = 256
    head_size = 8
    N = 2
    latent_dim = 128
    image_size = 224
    channels = 3
    seq_len = 197
    patch_size = 16
    lr = 3e-4
    dropout = 0.4
    batch_size = 8

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_data_df(path):
    with open(path, 'r') as f:
        data = json.load(f)
    image_paths_df = pd.DataFrame(data['images'])
    anno_cap_df = pd.DataFrame(data['annotations'])
    
    merged_df = anno_cap_df.merge(image_paths_df, left_on='image_id', right_on='id', how='left')
    ans_dict = merged_df.set_index('image_id')[['file_name', 'caption']].rename(columns={'file_name': 'image_path'}).to_dict()

    ans_df = pd.DataFrame(ans_dict)
    ans_df['image_path'] = np.random.permutation(ans_df['image_path'])
    ans_df['caption'] = np.random.permutation(ans_df['caption'])

    return ans_df


def get_sent(ds):
    yield ds['caption']


def get_tokenizer(ds):
    tokenizer_path = Path("tokenizer.json")
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<unk>", "<pad>", "<sos>", "<eos>"], min_frequency=2)
        tokenizer.train_from_iterator(get_sent(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


class SteganoDataset(Dataset):
    def __init__(self, image_dir, ans_df, tokenizer, seq_len, transforms=None):
        self.image_dir = image_dir
        self.image_paths = ans_df['image_path'].to_numpy()
        self.caption = ans_df['caption'].to_numpy()
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.sos = tokenizer.token_to_id("<sos>")
        self.eos = tokenizer.token_to_id("<eos>")
        self.pad = tokenizer.token_to_id("<pad>")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, inx):
        im = Image.open(os.path.join(self.image_dir, self.image_paths[inx])).convert('RGB')
        ca = self.caption[inx]
    
        if self.transforms is not None:
            im = self.transforms(im).half()  
    
        max_chars = (self.seq_len - 2) * 4  
        ca = ca[:max_chars] 
    
        ca = self.tokenizer.encode(ca).ids
        pad = self.seq_len - len(ca) - 1
        if pad < 0:
            raise ValueError(f"Sentence too long even after trimming: {ca}")
    
        dec_inp = torch.cat([
            torch.tensor([self.sos], dtype=torch.int64),
            torch.tensor(ca, dtype=torch.int64),
            torch.tensor([self.pad] * pad, dtype=torch.int64)
        ])
    
        label = torch.cat([
            torch.tensor(ca, dtype=torch.int64),
            torch.tensor([self.eos], dtype=torch.int64),
            torch.tensor([self.pad] * pad, dtype=torch.int64)
        ])
    
        return im, dec_inp, label


train_df = get_data_df(Paths.train_cap)
train_df = train_df.sample(frac=0.25, random_state=42)
tokenizer = get_tokenizer(train_df)

val_df = get_data_df(Paths.val_cap)
tokenizer_val = get_tokenizer(val_df)

transformers = T.Compose([
    T.Resize((Config.image_size, Config.image_size)),
    T.ToTensor()
])

def custom_collate_fn(batch):
    images, texts, labels = zip(*batch)

    images = torch.stack(images)  # Stack images
    texts = pad_sequence(texts, batch_first=True, padding_value=0) 
    labels = pad_sequence(labels, batch_first=True, padding_value=0) 

    return images, texts, labels

train_dataset = SteganoDataset(Paths.train_img, train_df, tokenizer, Config.seq_len, transformers)
val_dataset = SteganoDataset(Paths.val_img, val_df, tokenizer_val, Config.seq_len, transformers)


train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn, num_workers=2)


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=config.channels,
                              out_channels=config.d_model,
                              kernel_size=config.patch_size,
                              stride=config.patch_size)
        self.flatten = nn.Flatten(2)
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.cls = nn.Parameter(torch.randn((1, 1, config.d_model)), requires_grad=True)
        self.pos_embd = nn.Parameter(torch.randn((1, self.num_patches + 1, config.d_model)), requires_grad=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor):
        cls = self.cls.expand(x.shape[0], -1, -1)
        # (bs, in_c, h, w) --> (bs, emb_dim, new_h, new_w)
        x = self.conv(x)
        # (bs, emb_dim, new_h, new_w) --> (bs, emb_dim, new_h * new_w) --> (bs, new_h * new_w, emb_dim)
        x = self.flatten(x).permute(0, 2, 1)
        # (bs, new_h * new_w, emb_dim) --> (bs, num_patches + 1, emb_dim)
        x = torch.cat([x, cls], dim=1)
        # (bs, num_patches + 1, emb_dim) --> (bs, num_patches + 1, emb_dim)
        x += self.pos_embd
        # (bs, num_patches + 1, emb_dim)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_k = config.d_model // config.head_size
        self.head_size = config.head_size

        self.wq = nn.Linear(config.d_model, config.d_model)
        self.wk = nn.Linear(config.d_model, config.d_model)
        self.wv = nn.Linear(config.d_model, config.d_model)

        self.wo = nn.Linear(config.d_model, config.d_model)

    def forward(self, q, k, v, mask=None):
        b, s, d = q.shape
        q = self.wq(q).view(q.shape[0], q.shape[1], self.head_size, self.d_k).transpose(-2, -3)
        k = self.wk(k).view(k.shape[0], k.shape[1], self.head_size, self.d_k).transpose(-2, -3)
        v = self.wv(v).view(v.shape[0], v.shape[1], self.head_size, self.d_k).transpose(-2, -3)
        
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            att.masked_fill_(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = att @ v
        out = att.transpose(-2, -3).contiguous().view(b, s, d)
        out = self.wo(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.lin = nn.Linear(config.d_model, config.d_model * 4)
        self.out = nn.Linear(config.d_model * 4, config.d_model)

    def forward(self, x):
        x = self.lin(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class VaeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        
    def forward(self, x):
        res = x
        x = self.norm1(res + self.attention(x, x, x))
        res = x
        x = self.norm2(res + self.ff(x))
        return x


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = nn.ModuleList([VaeEncoder(config) for _ in range(config.N)])
        self.latent_dim = config.latent_dim
        self.fc_decoder = nn.Linear(self.latent_dim, config.d_model * (config.image_size // 16) * (config.image_size // 16))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, config.channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  
        )
        self.image_size = config.image_size

    def encode(self, x):
        for i in self.encoder:
            x = i(x)
        x = nn.Linear(x.shape[-1], self.latent_dim, device=x.device)(x)
        return x

    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(z.shape[0], -1, (self.image_size // 16), (self.image_size // 16))
        x = nn.ConvTranspose2d(x.shape[1], 64, kernel_size=4, stride=2, padding=1,device=x.device)(x)
        x = self.decoder_conv(x)
        return x


class SenderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prep_network = BertModel.from_pretrained("bert-base-uncased")
        self.img_emb = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.hidden_network = VAE(config)
        self.d_model = config.d_model

    def get_prep_emb(self, text):
        return self.prep_network(text)

    def get_img_emb(self, img):
        return self.img_emb(img)

    def concat_emb(self, text, img):
        t_emb = self.get_prep_emb(text).last_hidden_state.to(text.device)  
        i_emb = self.get_img_emb(img).last_hidden_state.to(img.device)   
        
        out = torch.cat([t_emb, i_emb], dim=-2)
        out = nn.Linear(out.shape[-1], self.d_model, device=out.device)(out) 
        return out

    def forward(self, text, img):
        x = self.concat_emb(text, img)
        out = self.hidden_network.encode(x)
        out = self.hidden_network.decode(out)
        return out


class ReciverModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn1 = Attention(config)
        self.attn2 = Attention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ln3 = nn.LayerNorm(config.d_model)
        self.patch_emb = PatchEmbedding(config)

        self.proj = nn.Linear(config.d_model, tokenizer.get_vocab_size())

    def forward(self, enc_out, dec_in, mask):
        enc_out = self.patch_emb(enc_out)
        x = dec_in
        x = self.ln1(x + self.attn1(dec_in, dec_in, dec_in))
        x = self.ln2(x + self.attn2(enc_out, enc_out, dec_in, mask))
        x = self.ln3(x + self.ff(x))
        return x

    def project(self, x):
        return self.proj(x)


embedding_layer = nn.Embedding(tokenizer.get_vocab_size(), Config.d_model, device=device).half()
loss_list = []
accuracy_list = []
all_preds = []
all_labels = []
all_ssim = []
all_psnr = []

def train_for_one_epoch(sender_model, reciver_model, train_dataloader, loss_fn_1, loss_fn_2, optimizer_1, optimizer_2, beta, device):
    sender_model.train()
    reciver_model.train()

    scaler = torch.cuda.amp.GradScaler()

    total_loss = 0
    correct = 0
    total = 0
    
    for image, text, label in (pbar := tqdm(train_dataloader, desc="Training")):
        image, text, label = image.to(device).half(), text.to(device), label.to(device)

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        with torch.cuda.amp.autocast():  
            image_pred = sender_model(text, image)
            loss1 = loss_fn_1(image, image_pred)

            mask = torch.tril(torch.ones(Config.seq_len, Config.seq_len, device=device)).half()
            text = embedding_layer(text)
            text_pred = reciver_model(image_pred, text, mask)
            text_pred = reciver_model.project(text_pred)

            loss2 = loss_fn_2(text_pred.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            loss = loss1 + beta * loss2
        
        scaler.scale(loss).backward()  
        scaler.step(optimizer_1)
        scaler.step(optimizer_2)
        scaler.update()

        total_loss += loss.item()

        predicted = text_pred.argmax(dim=-1)  
        correct += (predicted == label).sum().item()
        total += label.numel()

        all_preds.append(predicted.view(-1))
        all_labels.append(label.view(-1))


        ssim, psnr = calculate_image_metrics(image_pred, image)
        all_ssim.append(ssim)
        all_psnr.append(psnr)

        accuracy_list.append(correct/total)
        loss_list.append(loss.item())
        
        pbar.set_postfix(Loss=loss.item(), Accuracy=correct/total, SSIM=ssim, PSNR=psnr)

    accuracy = correct / total

    precision, recall, f1 = calculate_classification_metrics(torch.cat(all_preds), torch.cat(all_labels))

    avg_ssim = sum(all_ssim) / len(all_ssim)
    avg_psnr = sum(all_psnr) / len(all_psnr)

    print(f"Epoch Summary: Loss={total_loss:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, SSIM={avg_ssim:.4f}, PSNR={avg_psnr:.4f}")


config = Config()
sender_model = SenderModel(config).to(device)
reciver_model = ReciverModel(config).to(device)
loss_1 = nn.BCEWithLogitsLoss().to(device)
loss_2 = nn.CrossEntropyLoss().to(device)

optimizer_1 = optim.AdamW(sender_model.parameters(), lr=config.lr)
optimizer_2 = optim.AdamW(reciver_model.parameters(), lr=config.lr)


def calculate_classification_metrics(preds, labels):
    preds = preds.view(-1).cpu().numpy()  
    labels = labels.view(-1).cpu().numpy()

    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return precision, recall, f1

ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)

def calculate_image_metrics(image_pred, image_true):
    ssim = ssim_metric(image_pred, image_true).item()
    return ssim


train_for_one_epoch(sender_model, reciver_model, train_dataloader, loss_1, loss_2, optimizer_1, optimizer_2, 0.75, device)


torch.save(sender_model, 'sender.pt')
torch.save(reciver_model, 'reciver.pt')


precision, recall, f1 = calculate_classification_metrics(torch.cat(all_preds), torch.cat(all_labels))


ans = {"accuracy":accuracy_list[::300],
      "loss":loss_list[::300],
      "ssim":all_ssim[::300]}
ans = pd.DataFrame(ans)
ans.to_csv("result.csv")


import matplotlib.pyplot as plt
plt.plot(accuracy_list[::200])
plt.plot(loss_list[::200])
plt.plot(all_ssim[::200])
