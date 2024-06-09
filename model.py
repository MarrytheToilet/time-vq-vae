import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, channel, kernel_size=4, stride=stride, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.res_blocks(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channel, channel, kernel_size=4, stride=stride, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]
        )
        self.conv_out = nn.Conv1d(channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.res_blocks(x)
        x = self.conv_out(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, channel, n_res_channel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channel, n_res_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_res_channel, channel, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Quantize(nn.Module):
    def __init__(self, embed_dim, n_embed):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.embed = nn.Embedding(n_embed, embed_dim)
        self.embed.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

    def forward(self, x):
        flatten = x.reshape(-1, self.embed_dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed.weight.t()
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = torch.zeros(embed_ind.size(0), self.n_embed, device=x.device)
        embed_onehot.scatter_(1, embed_ind.view(-1, 1), 1)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        diff = (quantize - x).pow(2).mean()

        quantize = x + (quantize - x).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_ind):
        return self.embed(embed_ind)

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=1,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv1d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv1d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose1d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 2, 1)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        if dec_t.shape[2] > enc_b.shape[2]:
            dec_t = dec_t[:, :, :enc_b.shape[2]]
        else:
            enc_b = enc_b[:, :, :dec_t.shape[2]]

        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 2, 1)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        if upsample_t.shape[2] > quant_b.shape[2]:
            upsample_t = upsample_t[:, :, :quant_b.shape[2]]
        else:
            quant_b = quant_b[:, :, :upsample_t.shape[2]]

        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 2, 1)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 2, 1)

        dec = self.decode(quant_t, quant_b)

        return dec
