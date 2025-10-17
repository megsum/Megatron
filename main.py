import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files="discord_corpus.txt", vocab_size=8000, min_frequency=2)
tokenizer.save_model("tokenizer/")

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, n_layer=4, n_head=4, n_embd=256, block_size=128):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, n_embd)
        self.embed_positions = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=n_embd * 4,
                batch_first=True
            )
            for _ in range(n_layer)
        ])

        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, device=x.device).unsqueeze(0)
        x = self.embed_tokens(x) + self.embed_positions(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.lm_head(x)

class ChatDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=128):
        ids = tokenizer.encode(text).ids
        self.tokens = ids
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.block_size])
        y = torch.tensor(self.tokens[idx+1:idx+self.block_size+1])
        return x, y
    

def generate(prompt, max_new_tokens=50):
    tokens = tokenizer.encode(prompt).ids
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).cuda()
    for _ in range(max_new_tokens):
        logits = model(x)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
    return tokenizer.decode(x[0].tolist())

def main():
    model = TinyTransformer(vocab_size=8000).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    dataset = ChatDataset(
        text=open("discord_corpus.txt", "r", encoding="utf-8").read(),
        tokenizer=tokenizer,
        block_size=128
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(5):
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("loss:", loss.item())


if __name__ == "__main__":
    main()
