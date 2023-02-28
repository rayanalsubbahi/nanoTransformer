import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from DataLoader import TranslationDataset

#hyperparameters
batch_size = 64 #independent sequences to be processed in parallel
block_size = 64 #chunks of data to be processed (defining the context)
max_iters = 50000 #number of iterations to train for
eval_interval = 100 #evaluate the model every 100 iterations
lr = 1e-4 #learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device {device}')
eval_iters = 100 #number of iterations to evaluate for
n_embd = 384 #embedding dimension
n_heads = 4 #number of attention heads
n_layers = 3 #number of transformer layers
dropout = 0.2 #dropout rate

#referecnes
#https://github.com/hyunwoongko/transformer
#https://www.tensorflow.org/text/tutorials/transformer

#wget 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#load shakespare dataset
# with open('input.txt', 'r', encoding='utf-8') as f:
#     input_text = f.read()

# with open('target.txt', 'r', encoding='utf-8') as f:
#     target_text = f.read()
dataset = pd.read_csv('eng_-french.csv')
input = dataset['English words/sentences'].values
target = dataset['French words/sentences'].values

#concat all input vals
input_text = '\n'.join(input)
#concat all target vals
target_text = '\n'.join(target)

# #get the vocab of chars (unique chars occuring in the text)
input_chars = sorted(list(set(input_text)))
input_vocab_size = len(input_chars)

# #lookup tables mapping chars to integers and vice versa
input_stoi = {c:i for i,c in enumerate(input_chars)}
input_itos = {i:c for i,c in enumerate(input_chars)}
#add a padding token
input_stoi['<'] = input_vocab_size
input_itos[input_vocab_size] = '<'
input_pad_idx = input_vocab_size
input_vocab_size += 1
input_encoder = lambda s: [input_stoi[c] for c in s] #encode a string to a list of integers
input_decoder = lambda l: ''.join([input_itos[i] for i in l]) #decode a list of integers to a string

# #get the vocab of chars (unique chars occuring in the text) for target
target_chars = sorted(list(set(target_text)))
target_vocab_size = len(target_chars)

# #lookup tables mapping chars to integers and vice versa
target_stoi = {c:i for i,c in enumerate(target_chars)}
target_itos = {i:c for i,c in enumerate(target_chars)}
#add a padding token
target_stoi['<'] = target_vocab_size
target_itos[target_vocab_size] = '<'
target_pad_idx = target_vocab_size
target_vocab_size += 1
target_encoder = lambda s: [target_stoi[c] for c in s] #encode a string to a list of integers
target_decoder = lambda l: ''.join([target_itos[i] for i in l]) #decode a list of integers to a string

n = int(0.9*len(input))
train_loader = TranslationDataset(input[:n], target[:n], input_encoder, target_encoder, block_size, input_pad_idx, target_pad_idx)
val_loader = TranslationDataset(input[n:], target[n:], input_encoder, target_encoder, block_size, input_pad_idx, target_pad_idx)

# #get batches (data loading)
def get_batch(split):
    loader = train_loader if split == 'train' else val_loader
    #generate a small batch of of (x, y)
    ix = torch.randint(len(loader), (batch_size, ))
    x = torch.stack([loader[i][0] for i in ix])
    y = torch.stack([loader[i][2] for i in ix])
    shifted_y = torch.zeros_like(y)
    shifted_y[:, :-1] = y[:, 1:]
    x, y, shifted_y = x.to(device), y.to(device), shifted_y.to(device)
    src_mask = torch.stack([loader[i][1] for i in ix])
    trgt_mask = torch.stack([loader[i][3] for i in ix])
    src_mask, trgt_mask = src_mask.to(device), trgt_mask.to(device)
    return x, y, shifted_y, src_mask, trgt_mask

#xb, yb, shifted_yb = get_batch('train')
#decode a batch of sequences
# print('input:', input_decoder(xb[0, :].tolist()))
# print('target:', target_decoder(yb[0, :].tolist()))
# print('shifted target:', target_decoder(shifted_yb[0, :].tolist()))

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb, shifted_yb, src_mask, trgt_mask = get_batch(split)
            logits = model(xb, yb, src_mask, trgt_mask)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            shifted_yb = shifted_yb.view(B*T)
            loss = F.cross_entropy(logits, shifted_yb,ignore_index=target_pad_idx)
            losses[i] = loss
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    '''one head of self attention'''
    def __init__(self, head_size, mask=False):
        super().__init__()
        self.is_tril_mask=mask
        self.query = nn.Linear(n_embd, head_size, bias=False) 
        self.key = nn.Linear(n_embd, head_size, bias=False) 
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoded_x=None, mask=None):
        B, T, C = x.shape
        k = self.key(x) if encoded_x is None else self.key(encoded_x) # (B, T, C) 
        q = self.query(x) # (B, T, C)
        #compute attention scores (affinity between each token and each other token)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) ---> (B, T, T) 
        if mask is not None:
            mask = mask.unsqueeze(1) #(B, 1, T)
            wei = wei.masked_fill(mask == 0, float('-inf'))
        if self.is_tril_mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B, T, T)
        wei = F.softmax(wei, dim=-1) #(B, T, T)
        wei = self.dropout(wei) #(B, T, T)
        #perform weighted aggregation of values
        v = self.value(x) if encoded_x is None else self.value(encoded_x) #(B, T, C)
        out = wei @ v #(B, T, T) @ (B, T, C) ---> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    '''multiple heads of self attention in parallel'''
    def __init__(self, n_heads, head_size, mask=None):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, mask) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd) #project back to residual dimension
        self.dropout= nn.Dropout(dropout)

    def forward(self, x, encoded_x=None, mask=None):
        out = torch.cat([h(x, encoded_x, mask) for h in self.heads], dim=-1) #(B, T, C)
        out = self.proj(out) #(B, T, C)
        out = self.dropout(out) #(B, T, C)
        return out

class FeedForward(nn.Module):
    '''feed forward network (simple layer followed by non-linearity)'''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), #4* referes to the paper
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

#encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.multi_head_attention = MultiHeadAttention(n_heads, head_size, mask=False)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x, src_mask):
        x = x + self.multi_head_attention(x, mask=src_mask)
        x = self.ln1(x)
        x = x + self.ffwd(x)
        x = self.ln2(x)
        return x

#encoder model
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(input_vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([EncoderLayer(n_embd, n_heads) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_mask):
        B,T = x.shape
        #add token and position embeddings
        x = self.token_embedding_table(x) + self.position_embedding_table(torch.arange(T, device=device))
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
    
#decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.masked_multi_head_attention = MultiHeadAttention(n_heads, head_size, mask=True)
        self.cross_attention = MultiHeadAttention(n_heads, head_size, mask=False)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x, encoded_x, src_mask, trgt_mask):
        x = x + self.masked_multi_head_attention(x, mask=trgt_mask)
        x = self.ln1(x)
        x = x + self.cross_attention(x, encoded_x, mask=src_mask)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        x = self.ln3(x)
        return x


#decoder model
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(target_vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.layers = nn.ModuleList([DecoderLayer(n_embd, n_heads) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, y, encoded_x, src_mask, trgt_mask):
        B,T = y.shape
        #add token and position embeddings
        y = self.token_embedding_table(y) + self.position_embedding_table(torch.arange(T, device=device))
        y = self.dropout(y)
        for layer in self.layers:
            y = layer(y, encoded_x, src_mask, trgt_mask)
        return y

#transformer model
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.lm_head = nn.Linear(n_embd, target_vocab_size)
    
    def forward(self, x, y, src_mask, tgt_mask):
        encoded_x = self.encoder(x, src_mask)
        decoded_y = self.decoder(y, encoded_x, src_mask, tgt_mask)
        logits = self.lm_head(decoded_y)
        return logits

    def generate(self, x, idx, max_new_tokens):
        #idx is current context (B, T)
        for _ in range(max_new_tokens):
            #crop the context to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B, T)
            #get the predictions
            logits  = self(x, idx_cond, None, None) # (B, T, C)
            #get last time step
            logits = logits[:, -1, :] # (B, C)
            #get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = Transformer()
m = model.to(device)

#get number of parameters
nbParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'number of parameters: {nbParams}')

#create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

#training loop
for iter in range(max_iters): 

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'iter {iter} | train loss {losses["train"]:.3f} | val loss {losses["val"]:.3f}')
        
    #sample a batch of data
    xb, yb, shifted_yb, src_mask, trgt_mask = get_batch('train')

    #forward pass
    logits = m(xb, yb, src_mask, trgt_mask)

    #compute loss
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    shifted_yb = shifted_yb.view(B*T)
    #loss
    loss = F.cross_entropy(logits, shifted_yb, ignore_index=target_pad_idx)

    #backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
batch_size = 1
input_sentence = 'I love to sing'
input_sentence = torch.tensor(input_encoder(input_sentence), device=device)
#reshape to (B, T)
input_sentence = input_sentence.view(1, -1)
context = torch.ones((1,1), dtype=torch.long, device=device)
print(target_decoder(m.generate(input_sentence, context, max_new_tokens=30)[0].tolist()))
