"""### Importing Libraries"""

import torch
import torch.nn as nn
import numpy as np
import math
from torchinfo import summary
import sentencepiece as spm
import tempfile
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import torch.optim as optim
import sacrebleu
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.optim import AdamW

"""## Transformer

### MultiHead Attention
"""

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.head_dim = embedding_dim // num_heads    # Calculating dimension per head
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)  # Linear or WeightMatrices i.e Wq,... projections for Query, Key, Value, Input shape: (batch_size, seq_len, embedding_dim),Output shape: (batch_size, seq_len, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        self.output_proj = nn.Linear(embedding_dim, embedding_dim)    # Final output projection

        self.dropout = nn.Dropout(dropout)         # Dropout for regularization

    def forward(self, query_input, key_input, value_input, mask=None):

        batch_size = query_input.size(0)

        query = self.query_proj(query_input).view(                                        # Applying linear Projections and Reshaping for multihead attn from , Input shape: (batch_size, seq_len, embedding_dim)
            batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)               # to (batch_size, num_heads, seq_len, head_dim)
        key = self.key_proj(key_input).view(
            batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value_input).view(
            batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)    # calculating attn scores and applying and scaling,Shape: (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))                       # applying mask for causal masking setting to -inf so becomes zero after softmax

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)     # appying softmax,Shape: (batch_size, num_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)                             # multipying with value ,shape(batch_size, num_heads, seq_len, head_dim)

        output = output.transpose(1, 2).contiguous().view(                          # Reshaping and applying final projection,Reshape from (batch_size, num_heads, seq_len, head_dim) to batch_size, seq_len, embedding_dim)
            batch_size, -1, self.embedding_dim)

        return self.output_proj(output)                                              # Final output shape: (batch_size, seq_len, embedding_dim)

"""### Residual Connections + Dropout

"""

class ResidualConnection(nn.Module):
    def __init__(self, size, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)      # dropout
        self.layer_norm = nn.LayerNorm(size)           # layer norm

    def forward(self, input_tensor: torch.Tensor, sublayer) -> torch.Tensor:
        normalized = self.layer_norm(input_tensor)
        sublayer_output = sublayer(normalized)          # apply any sublayer we'll pass ffn or mha
        dropped = self.dropout(sublayer_output)
        return input_tensor + dropped

"""### Pos Embeddings"""

def get_positional_encoding(max_len, d_emb):

    pos = np.arange(max_len)[:, np.newaxis]     # Generate position indices [0, 1, 2, ...max_len-1] and reshape to (max_len, 1)
    i = np.arange(d_emb)[np.newaxis, :]     # Generate dimension indices [0, 1, 2, ...d_emb-1] and reshape to (1, d_emb)

    angles = pos / np.power(10000, 2 * i / d_emb)  # calculate angle based on formula mentioned in paper

    positional_encoding = np.zeros((max_len, d_emb)) # output array

    # Applying sine to even indices and cosine to odd indices
    positional_encoding[:, ::2] = np.sin(angles[:, ::2])    # even indices
    positional_encoding[:, 1::2] = np.cos(angles[:, 1::2])  # odd indices

    return positional_encoding[np.newaxis, ...] # adding batch dim and returning

"""### Single Encoder Block"""

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadedAttention(                    # Attention layer
            num_heads=config.num_attention_heads,
            embedding_dim=config.d_embed,
            dropout=config.dropout
        )
        self.ffn = nn.Sequential(                                        # ffn
            nn.Linear(config.d_embed, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_embed)
        )
        self.skip1 = ResidualConnection(config.d_embed, config.dropout)                    # Skip connections
        self.skip2 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, inputs, mask=None):
        # Applying self-attention with skip connection
        attended = self.skip1(inputs, lambda x: self.attention(x, x, x, mask=mask))
        # Applying feed forward with skip connection
        output = self.skip2(attended, self.ffn)
        return output

"""### Complete Encoder"""

class Encoder(nn.Module):
    # Steps: 1. Token embedding
     #      2. Add fixed positional encoding
      #     3. Process through encoder blocks,ffn
       #    4. Apply final layer normalization

    def __init__(self, config):
        super().__init__()
        self.embed_size = config.d_embed
        self.word_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        pos_encoding = get_positional_encoding(config.max_seq_len, config.d_embed)          # Creating fixed positional encoding
        self.register_buffer('pos_encoding', torch.FloatTensor(pos_encoding))
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.N_encoder)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, tokens, mask=None):
        word_vectors = self.word_embed(tokens)            # Converting tokens to embeddings
        pos_vectors = self.pos_encoding[:, :word_vectors.size(1), :]         # Get position encoding for our sequence length
        combined = self.dropout(word_vectors + pos_vectors)          # Combine both word embeddings and positional encoding
        # Passing through encoder blocks stacked
        for block in self.blocks:
            combined = block(combined, mask)
        # Final layer norm
        return self.norm(combined)

"""### Single Decoder Block"""

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.masked_self_attention = MultiHeadedAttention(config.num_attention_heads, config.d_embed)
        self.cross_attention = MultiHeadedAttention(config.num_attention_heads, config.d_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.feedforward_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_embed)
        )
        self.residuals = nn.ModuleList([ResidualConnection(config.d_embed, config.dropout)
                                       for i in range(3)])

    def forward(self, encoder_output, encoder_mask, decoder_input, decoder_mask):
        decoder_state = self.residuals[0](decoder_input,
            lambda x: self.masked_self_attention(x, x, x, mask=decoder_mask))             # 1st sub-layer= masked self-attention

        decoder_state = self.residuals[1](decoder_state,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, mask=encoder_mask))         # 2nd sub-layer= cross-attention with encoder output

        # 3rd sub-layer: ffn
        decoder_state = self.residuals[2](decoder_state, self.feed_forward)
        return decoder_state

"""### Complete Decoder Block"""

class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.d_embed
        self.token_embedding = nn.Embedding(config.decoder_vocab_size, config.d_embed)
        positional_encodings = get_positional_encoding(config.max_seq_len, config.d_embed)
        self.register_buffer('positional_encodings', torch.FloatTensor(positional_encodings))
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.decoder_layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.N_decoder)])
        self.layer_norm = nn.LayerNorm(config.d_embed)
        self.output_projection = nn.Linear(config.d_embed, config.decoder_vocab_size)

    def future_mask(self, sequence_length):
        causal_mask = (torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)!=0)
        device = next(self.parameters()).device
        return causal_mask.unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, encoder_output, encoder_mask, target_tokens, target_padding_mask):
        sequence_length = target_tokens.size(1)


        token_embeddings = self.token_embedding(target_tokens)
        position_encoded = token_embeddings + self.positional_encodings[:, :sequence_length, :]         # Token embeddings and positional encoding
        decoder_state = self.embedding_dropout(position_encoded)


        future_mask = self.future_mask(sequence_length)
        future_mask = future_mask.expand(target_tokens.size(0), -1, sequence_length, sequence_length)         # Generate future mask and combine with padding mask

        if target_padding_mask.size(-1) != sequence_length:
            target_padding_mask = (target_tokens == 0).unsqueeze(1).unsqueeze(2)

        # Now expand to [batch_size, 1, seq_len, seq_len]
        target_padding_mask = target_padding_mask.expand(-1, -1, sequence_length, sequence_length)
        attention_mask = target_padding_mask | future_mask

        # Passing through decoder layers
        for decoder_layer in self.decoder_layers:
            decoder_state = decoder_layer(encoder_output, encoder_mask, decoder_state, attention_mask)

        normalized_output = self.layer_norm(decoder_state)
        logits = self.output_projection(normalized_output)
        return logits

"""### Combining all prev Stuff here"""

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source_tokens, source_mask, target_tokens, target_mask):
        encoder_output = self.encoder(source_tokens, source_mask) # a run thrrough encoder block
        # Then through decoder with encoder output
        return self.decoder(encoder_output, source_mask, target_tokens, target_mask)  # a run through decoder blocks with encoder input

"""### Initializing Model"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model config
class Config:
    def __init__(self):
        self.d_embed = 512  # Embedding dimension
        self.feedforward_dim = 256    # Feed-forward hidden layer size
        self.num_attention_heads = 8    # Number of attention heads
        self.N_encoder = 6  # Number of encoder layers
        self.N_decoder = 6  # Number of decoder layers
        self.dropout = 0.1  # Dropout rate
        self.max_seq_len = 512  # Maximum sequence length
        self.encoder_vocab_size = 30000  # Source vocabulary size
        self.decoder_vocab_size = 30000  # Target vocabulary size



def initialize_transformer(device=None,d_embed=512,feedforward_dim=2048,num_attention_heads=8,n_encoder=6,n_decoder=6,dropout=0.1,max_seq_len=512,encoder_vocab_size=30000,decoder_vocab_size=30000):

    if device is None:
        device = DEVICE

    # Create custom config with provided parameters
    config = Config()
    config.d_embed = d_embed
    config.feedforward_dim = feedforward_dim
    config.num_attention_heads = num_attention_heads
    config.N_encoder = n_encoder
    config.N_decoder = n_decoder
    config.dropout = dropout
    config.max_seq_len = max_seq_len
    config.encoder_vocab_size = encoder_vocab_size
    config.decoder_vocab_size = decoder_vocab_size

    # Create encoder and decoder
    encoder = Encoder(config)
    decoder = Decoder(config)

    # Initialize the full transformer model
    model = Transformer(encoder, decoder)
    return model.to(device)  # Move model to GPU if available

# def test_transformer_forward():
#     """
#     Create a dummy forward pass to test the transformer model
#     """
#     # Initialize a small transformer for testing
#     model = initialize_transformer(
#         d_embed=64,
#         feedforward_dim=256,
#         num_attention_heads=4,
#         n_encoder=2,
#         n_decoder=2,
#         max_seq_len=32,
#         encoder_vocab_size=100,
#         decoder_vocab_size=100
#     )

#     # Create dummy batch
#     batch_size = 2
#     seq_length = 10

#     # Create random input tokens
#     source_tokens = torch.randint(1, 100, (batch_size, seq_length))
#     target_tokens = torch.randint(1, 100, (batch_size, seq_length))

#     # Create masks (1 for valid tokens, 0 for padding)
#     source_mask = torch.ones(batch_size, 1, 1, seq_length).bool()
#     target_mask = torch.ones(batch_size, 1, seq_length, seq_length).bool()

#     # Move everything to device
#     source_tokens = source_tokens.to(DEVICE)
#     source_mask = source_mask.to(DEVICE)
#     target_tokens = target_tokens.to(DEVICE)
#     target_mask = target_mask.to(DEVICE)

#     # Forward pass
#     print("Input shapes:")
#     print(f"Source tokens: {source_tokens.shape}")
#     print(f"Source mask: {source_mask.shape}")
#     print(f"Target tokens: {target_tokens.shape}")
#     print(f"Target mask: {target_mask.shape}")

#     # Run model in eval mode
#     model.eval()
#     with torch.no_grad():
#         output = model(source_tokens, source_mask, target_tokens, target_mask)

#     print("\nOutput shape:", output.shape)
#     print("Forward pass successful!")
#     return output

# if __name__ == "__main__":
# #     test_transformer_forward()
#       print("hello")

"""### Loading and Splitting the Data"""

# # Load the dataset
# df = pd.read_csv("spoc_train.csv")

# # 1. Single line translations
# single_lines = [(str(row['text']) if pd.notna(row['text']) else "",
#                 str(row['code']) if pd.notna(row['code']) else "")
#                 for _, row in df.iterrows()]

# # 2. Full program translations
# problems = defaultdict(lambda: {"text": [], "code": []})
# for _, row in df.iterrows():
#     text = str(row['text']) if pd.notna(row['text']) else ""
#     code = str(row['code']) if pd.notna(row['code']) else ""
#     problems[row['probid']]["text"].append(text)
#     problems[row['probid']]["code"].append(code)

# full_programs = [(
#     "\n".join(data["text"]),  # Join pseudocode lines
#     "\n".join(data["code"])   # Join code lines
# ) for probid, data in problems.items()]

# # Combine both datasets
# combined_tuples = single_lines + full_programs

# # Randomly split the data
# np.random.seed(42)
# indices = np.random.permutation(len(combined_tuples))

# # Calculate split sizes
# total_size = len(combined_tuples)
# train_size = int(0.7 * total_size)
# val_size = int(0.15 * total_size)

# # Split indices
# train_indices = indices[:train_size]
# val_indices = indices[train_size:train_size + val_size]
# test_indices = indices[train_size + val_size:]

# # Create splits using the random indices
# train_tuples = [combined_tuples[i] for i in train_indices]
# val_tuples = [combined_tuples[i] for i in val_indices]
# test_tuples = [combined_tuples[i] for i in test_indices]

# print(f"Dataset splits - Train: {len(train_tuples)}, Validation: {len(val_tuples)}, Test: {len(test_tuples)}")

SRC = "text"  # Source is pseudocode text
TRG = "code"  # Target is C++ code
src_vocab_size = 8000
tgt_vocab_size = 8000
vocab_sizes = {"text": src_vocab_size, "code": tgt_vocab_size}
max_seq_len = 100

vocab_size = 4000

# Some Global Variables
PAD, UNK, BOS, EOS = 0, 1, 2, 3

"""### Sentence Piece Tokenizer"""

# Creating temp files for training data
def write_sentences_to_file(sentences):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
        return f.name

# # Separate text and code
# text_sequences = [pair[0] for pair in train_tuples]  # Pseudocode text
# code_sequences = [pair[1] for pair in train_tuples]  # C++ code

# # Write sequences to temporary files
# text_input_file = write_sentences_to_file(text_sequences)
# code_input_file = write_sentences_to_file(code_sequences)

# # Train text tokenizer model for pseudocode
# text_sp = spm.SentencePieceProcessor()
# spm.SentencePieceTrainer.train(input=text_input_file,model_prefix='text_tokenizer',vocab_size=vocab_size,character_coverage=1.0,model_type='bpe',pad_id=0,unk_id=1,bos_id=2,eos_id=3)

# # Train code tokenizer model for C++
# code_sp = spm.SentencePieceProcessor()
# spm.SentencePieceTrainer.train(input=code_input_file,model_prefix='code_tokenizer',vocab_size=vocab_size,character_coverage=1.0,model_type='bpe',pad_id=0,unk_id=1,bos_id=2,eos_id=3)

# # Load the trained models
# text_sp.load('text_tokenizer.model')
# code_sp.load('code_tokenizer.model')

# # Create dictionaries for easy access
# tokenizers = {"text": text_sp.encode_as_ids,"code": code_sp.encode_as_ids}
# detokenizers = {"text": text_sp.decode_ids,"code": code_sp.decode_ids}
# id_to_pieces = {"text": text_sp.id_to_piece,"code": code_sp.id_to_piece}

"""### Testing if it works"""

# # Test pseudocode tokenization
# text_example = train_tuples[0][0]  # First pseudocode text
# print(f"\nPseudocode text: {text_example}")
# tokenized_text = tokenizers[SRC](text_example)
# print(f"Tokenized: {tokenized_text}")
# detokenized_text = detokenizers[SRC](tokenized_text)
# print(f"Detokenized: {detokenized_text}")

# # Test code tokenization
# code_example = train_tuples[0][1]  # First C++ code
# print(f"\nC++ code: {code_example}")
# tokenized_code = tokenizers[TRG](code_example)
# print(f"Tokenized: {tokenized_code}")
# detokenized_code = detokenizers[TRG](tokenized_code)
# print(f"Detokenized: {detokenized_code}")

# # Show sample tokens for both tokenizers
# print("\nSample pseudocode tokens:")
# print([id_to_pieces[SRC](i) for i in range(20)])
# print("\nSample C++ code tokens:")
# print([id_to_pieces[TRG](i) for i in range(20)])

"""### Getting Data Ready For Training - UTILITY FUNCTIONS"""

def tokenize_dataset(dataset):   # tokenizing the dataset and adding bos and eos tokens
    return [(torch.tensor([BOS] + tokenizers[SRC](src_text)[0:max_seq_len-2] + [EOS]),
             torch.tensor([BOS] + tokenizers[TRG](trg_text)[0:max_seq_len-2] + [EOS]))
            for src_text, trg_text in dataset]

class SpocDataset(Dataset):            # class for dataset
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_mask(tensor):
    return (tensor == PAD).unsqueeze(1).unsqueeze(2)     #creating padding mask

def pad_sequence(batch):          # padding sequences to same length and creating attn masks
    src_seqs = [src for src, trg in batch]
    trg_seqs = [trg for src, trg in batch]

    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=PAD)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_seqs, batch_first=True, padding_value=PAD)

    # Create padding masks
    src_mask = create_mask(src_padded)
    trg_mask = create_mask(trg_padded)

    return {
        'src': src_padded,
        'trg': trg_padded,
        'src_mask': src_mask,
        'trg_mask': trg_mask
    }

class Dataloaders: # dataloaders for training
    def __init__(self, train_tuples, val_tuples, test_tuples, batch_size=64):
        # Create datasets
        train_dataset = SpocDataset(tokenize_dataset(train_tuples))
        valid_dataset = SpocDataset(tokenize_dataset(val_tuples))
        test_dataset = SpocDataset(tokenize_dataset(test_tuples))

        # Create dataloaders with padding
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pad_sequence
        )

        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_sequence
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_sequence
        )

"""### Training Functions"""

def format_transformer_inputs(x, y): # prepare input tensors for training
    src = x.to(DEVICE)

    trg_in = y[:, :-1].to(DEVICE) # decoder i/p all tokens except last

    trg_out = y[:, 1:].contiguous().view(-1).to(DEVICE) # except first all tokens

    src_pad_mask = (src == PAD).unsqueeze(1).unsqueeze(2) # attn mask for src

    trg_pad_mask = (trg_in == PAD).unsqueeze(1).unsqueeze(2) # attn mask for target

    return src, trg_in, trg_out, src_pad_mask, trg_pad_mask

def generatecode(model, x):        # for translations using trained model
    with torch.no_grad():
        dB = x.size(0)
        y = torch.tensor([[BOS]*dB]).view(dB, 1).to(DEVICE)
        x_pad_mask = (x == PAD).view(x.size(0), 1, 1, x.size(-1)).to(DEVICE)
        memory = model.encoder(x, x_pad_mask)
        for i in range(max_seq_len):
            y_pad_mask = (y == PAD).view(y.size(0), 1, 1, y.size(-1)).to(DEVICE)
            logits = model.decoder(memory, x_pad_mask, y, y_pad_mask)
            last_output = logits.argmax(-1)[:, -1]
            last_output = last_output.view(dB, 1)
            y = torch.cat((y, last_output), 1).to(DEVICE)
    return y

def remove_pad(sent):                     # removing padding and eos tokens
    if sent.count(EOS)>0:
      sent = sent[0:sent.index(EOS)+1]
    while sent and sent[-1] == PAD:
            sent = sent[:-1]
    return sent

def decode_sentence(detokenizer, sentence_ids):       # tokens to text
    if not isinstance(sentence_ids, list):
        sentence_ids = sentence_ids.tolist()
    sentence_ids = remove_pad(sentence_ids)
    return detokenizer(sentence_ids).replace("<bos>", "")\
           .replace("<eos>", "").strip().replace(" .", ".")

def validate(model, dataloader, loss_fn):       # computing validation loss
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(DEVICE)
            trg = batch['trg'].to(DEVICE)
            src_mask = batch['src_mask'].to(DEVICE)
            trg_mask = batch['trg_mask'].to(DEVICE)

            # Prepare target inputs and outputs
            trg_input = trg[:, :-1]  # all but last token
            trg_output = trg[:, 1:].contiguous().view(-1)  # all but first token, flattened

            # Forward pass
            pred = model(src, src_mask, trg_input, trg_mask)
            pred = pred.view(-1, pred.size(-1))
            losses.append(loss_fn(pred, trg_output).item())
    return np.mean(losses)

def evaluate(model, dataloader, num_batch=None):        #bleu score for evaluation
    model.eval()
    refs, cans, bleus = [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            src = batch['src'].to(DEVICE)
            trg = batch['trg'].to(DEVICE)
            src_mask = batch['src_mask'].to(DEVICE)
            trg_mask = batch['trg_mask'].to(DEVICE)

            translation = generatecode(model, src)

            refs = refs + [decode_sentence(detokenizers[TRG], trg[i]) for i in range(len(src))]
            cans = cans + [decode_sentence(detokenizers[TRG], translation[i]) for i in range(len(src))]
            if num_batch and idx>=num_batch:
                break
        bleus.append(sacrebleu.corpus_bleu(cans, [refs]).score)

        for i in range(min(3, len(src))):   # printing some examples
            print(f'src:  {decode_sentence(detokenizers[SRC], src[i])}')
            print(f'trg:  {decode_sentence(detokenizers[TRG], trg[i])}')
            print(f'pred: {decode_sentence(detokenizers[TRG], translation[i])}')
        return np.mean(bleus)

def train(model, dataloaders, epochs=10):         # training func
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    grad_norm_clip = 1.0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        num_batches = len(dataloaders.train_loader)
        pbar = tqdm(enumerate(dataloaders.train_loader), total=num_batches)

        for idx, batch in pbar:
            optimizer.zero_grad()
            src = batch['src'].to(DEVICE)
            trg = batch['trg'].to(DEVICE)
            src_mask = batch['src_mask'].to(DEVICE)
            trg_mask = batch['trg_mask'].to(DEVICE)

            # Prepare target inputs and outputs
            trg_input = trg[:, :-1]  # all but last token
            trg_output = trg[:, 1:].contiguous().view(-1)  # all but first token, flattened

            # Forward pass
            pred = model(src, src_mask, trg_input, trg_mask)
            pred = pred.view(-1, pred.size(-1))

            # Calculate loss
            loss = loss_fn(pred, trg_output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

            if idx > 0 and idx % 50 == 0:
                pbar.set_description(f'train loss={loss.item():.3f}, lr={scheduler.get_last_lr()[0]:.5f}')

        train_loss = np.mean(train_losses)
        valid_loss = validate(model, dataloaders.valid_loader, loss_fn)
        print(f'Epoch {epoch}: train_loss={train_loss:.5f}, valid_loss={valid_loss:.5f}')

    return train_loss, valid_loss

# """### Training"""

# print("Creating dataloaders...")
# dataloaders = Dataloaders(train_tuples, val_tuples, test_tuples, batch_size=128)
# print("Initializing model...")
# model = initialize_transformer(d_embed=256,
#                             feedforward_dim=512,
#                             num_attention_heads=8,
#                             n_encoder=4,
#                             n_decoder=4,
#                             dropout=0.1,
#                             max_seq_len=max_seq_len,
#                             encoder_vocab_size=vocab_sizes[SRC],
#                             decoder_vocab_size=vocab_sizes[TRG])
# print("Starting Training...")

# train_loss, valid_loss = train(model, dataloaders, epochs=7)

# # Save Model
# torch.save(model.state_dict(), f'psd_code_transformer.pt')
# # # Evaluate on all splits
# # print("\nEvaluating model...")
# # print("Train set examples:")
# # train_bleu = evaluate(model, dataloaders.train_loader, num_batch=20)
# # print("\nValidation set examples:")
# # valid_bleu = evaluate(model, dataloaders.valid_loader)
# # print("\nTest set examples:")
# # test_bleu = evaluate(model, dataloaders.test_loader)

# # Save results
# results = {
#     'model_stats': {
#         'train_loss': float(train_loss),
#         'valid_loss': float(valid_loss),
#         # 'train_bleu': float(train_bleu),
#         # 'valid_bleu': float(valid_bleu),
#         # 'test_bleu': float(test_bleu)
#     }
# }

# # Save results
# torch.save(results, 'training_results.pt')

# print(f'\nFinal Results:')
# print(f'Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
# print(f'BLEU Scores:')
# print(f'Train: {train_bleu:.2f}, Valid: {valid_bleu:.2f}, Test: {test_bleu:.2f}')

# Functions to Create Streamlit
import os
def load_translation_model(model_path=None, text_spPath=None, code_spPath=None):
    global text_sp, code_sp, tokenizers, detokenizers
    print(model_path)
    # Load tokenizers if not already loaded

    print("Loading tokenizers...")
    text_sp = spm.SentencePieceProcessor()
    code_sp = spm.SentencePieceProcessor()
    text_sp.load(text_spPath)
    code_sp.load(code_spPath)

    # Update tokenizer dictionaries
    tokenizers = {"text": text_sp.encode_as_ids, "code": code_sp.encode_as_ids}
    detokenizers = {"text": text_sp.decode_ids, "code": code_sp.decode_ids}

    # Initialize model
    print("Initializing model...")
    model = initialize_transformer(
        d_embed=256,
        feedforward_dim=512,
        num_attention_heads=8,
        n_encoder=4,
        n_decoder=4,
        dropout=0.1,
        max_seq_len=max_seq_len,
        encoder_vocab_size=vocab_sizes[SRC],
        decoder_vocab_size=vocab_sizes[TRG]
    )

    # Load saved weights if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model = model.to(DEVICE)
    model.eval()
    return model

def psuedocodetocode(model, arabic_text):

    # If model is not provided, load it
    if model is None:
        model = load_translation_model()

    model.eval()
    with torch.no_grad():
        # Tokenize input text
        src_tokens = torch.tensor([[BOS] + tokenizers[SRC](arabic_text) + [EOS]]).to(DEVICE)

        # Create source padding mask
        src_mask = (src_tokens == PAD).unsqueeze(1).unsqueeze(2).to(DEVICE)

        # Get translation
        translation = generatecode(model, src_tokens)

        # Decode translation
        english_text = decode_sentence(detokenizers[TRG], translation[0])

    return english_text

# model = load_translation_model('path/to/model/weights.pt')  # Load once
# psudocode =  "print 'hello'"
# code = psuedocodetocode(model, psudocode)
# print(f"Pseudocode: {psudocode}")
# print(f"C++: {code}")
# psudocode =  'if x is greater than 10 then print x'
# code = psuedocodetocode(model, psudocode)
# print(f"Pseudocode: {psudocode}")
# print(f"C++: {code}")
# psudocode =  """set arr to [1, 2, 3, 4, 5]
# for each num in arr do print num
# """
# code = psuedocodetocode(model, psudocode)
# print(f"Pseudocode: {psudocode}")
# print(f"C++: {code}")