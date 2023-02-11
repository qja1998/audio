import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import distributed as dist_fn

import distributed as dist_fn
class Model(nn.Module):
    def __init__(self, in_channels=1,
                 num_hiddens=128,
                 num_residual_layers=2,
                 num_residual_hiddens=32,
                 embedding_dim=64,
                 num_embeddings=512,
                 commitment_cost=0.25):
        super(Model, self).__init__()
        
        self.encoder_bot = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, stride=4)#4
        self.pre_vq_conv_bot = nn.Conv2d(in_channels=num_hiddens,
                                     out_channels=embedding_dim,
                                     kernel_size=1, stride=1)
        self.encoder_top = Encoder(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens, stride=2)#2
        self.pre_vq_conv_top = nn.Conv2d(in_channels=embedding_dim + num_hiddens,
                                     out_channels=embedding_dim,
                                     kernel_size=1, stride=1)
        self.vq_top = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)
        self.decoder_top = Decoder(embedding_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, stride=2)#2
        self.vq_bot = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)
        
        self.upsample_t = nn.ConvTranspose2d(
            embedding_dim, embedding_dim, 4, stride=2, padding=1
        )
        self.decoder = Decoder(embedding_dim + embedding_dim, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, stride=4)#4

    def encode(self, x):
        # torch.Size([1, 128, 20, 256])
        # torch.Size([1, 128, 20, 256]) torch.Size([1, 128, 10, 128])
        # torch.Size([1, 128, 20, 256]) torch.Size([1, 64, 10, 128])
        # torch.Size([1, 128, 20, 256]) torch.Size([1, 64, 10, 128]) torch.Size([1, 64, 10, 128])
        # torch.Size([1, 64, 20, 256]) torch.Size([1, 64, 10, 128])
        enc_b = self.encoder_bot(x)         # 1 -> 128
        enc_t = self.encoder_top(enc_b)     # 128 -> 128
        enc_t = self.pre_vq_conv_bot(enc_t) # 128 -> 64

        diff_t, quantized_t, perplexity_t, id_t = self.vq_top(enc_t) # 64 -> 64
        diff_t = diff_t.unsqueeze(0)
        dec_t = self.decoder_top(quantized_t) # 64 -> 64
        #print(dec_t.shape, enc_b.shape)
        enc_b = torch.cat([dec_t, enc_b], 1)  # 64 + 128
        enc_b = self.pre_vq_conv_top(enc_b)   # 64 + 128 -> 64

        diff_b, quantized_b, perplexity_b, id_b = self.vq_bot(enc_b) # 64 -> 64
        diff_b = diff_b.unsqueeze(0)

        return (quantized_t, quantized_b), torch.mean(diff_t + diff_b), torch.mean(perplexity_t + perplexity_b), (id_t, id_b)

    def decode(self, quantized_t, quantized_b):
        #print(quantized_t.shape, quantized_b.shape)
        #upsample_t = self.upsample_t(quantized_t)
        upsample_t = self.upsample_t(quantized_t)
        quantized = torch.cat([upsample_t, quantized_b], 1)
        dec = self.decoder(quantized)

        return dec

    def reconstruction_loss(self, x, y):
        return torch.mean((x - y) ** 2)
    
    def forward(self, x):
        quantized, loss, preplexity, _ = self.encode(x)
        quantized_t, quantized_b = quantized
        recon = self.decode(quantized_t, quantized_b)
        
        return loss, recon, preplexity
    
    def latent_to_out(self, latent_t, latent_b):
        quant_t = self.vq_top.embed_code(latent_t)
        quant_t = quant_t.permute(0, 3, 1, 2).contiguous()
        quant_b = self.vq_bot.embed_code(latent_b)
        quant_b = quant_b.permute(0, 3, 1, 2).contiguous()

        dec = self.decode(quant_t, quant_b).detach().cpu().numpy()
        return dec
    
class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay=0.99, eps=1e-5):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        
        #self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        #self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1/self.num_embeddings)
        
        # optimizer�? ?��?��?��?�� ?���? ?��?��
        # state_dict�? ?��?�� �??��
        embed = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())
        
    def embedding_encode(self, z):
        z_shpae = z.size()
        encoding_indices = torch.argmin(z, dim=1)
        encodings = torch.zeros(z_shpae[0], self.num_embeddings, device=z.device)
        # sparse
        encodings = torch.index_select(self.embed, dim=0, index=encoding_indices)

        # nuflatten
        quantized = encodings.view(z_shpae)

        return quantized

    def forward(self, x):
        # (n, c, h, w) -> (n, h, w, c)
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.size()
        x_flat = x.view(-1, self.embedding_dim)
        # trick
        distance = (
            x_flat.pow(2).sum(1, keepdim=True)
            - 2 * x_flat @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        # nearest neighborhood look-up     
        _, embed_ind = (-distance).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_embeddings).type(x_flat.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantized = self.embed_code(embed_ind)
        
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = x_flat.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_embeddings * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        
        e_latent_loss = F.mse_loss(quantized.detach(), x) # codebook learning
        q_latent_loss = F.mse_loss(quantized, x.detach()) # encoder learning
        loss = q_latent_loss + e_latent_loss * self.commitment_cost
        
        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(embed_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, embed_ind
    
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self.block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([ResidualBlock(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, stride):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens//2,
                                kernel_size=4,
                                stride=2, padding=1)
        nn.init.kaiming_uniform_(self.conv_1.weight, nonlinearity='relu')
        self.res_1 = ResidualBlock(num_hiddens//2, num_hiddens//2, num_hiddens//8)

        if stride == 4:
            self.conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                    out_channels=num_hiddens,
                                    kernel_size=4,
                                    stride=2, padding=1)
            nn.init.kaiming_uniform_(self.conv_2.weight, nonlinearity='relu')
            self.res_2 = ResidualBlock(num_hiddens, num_hiddens, num_hiddens//4)

            self.conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                    out_channels=num_hiddens,
                                    kernel_size=3,
                                    stride=1, padding=1)
            nn.init.kaiming_uniform_(self.conv_3.weight, nonlinearity='relu')
            
            self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                                num_hiddens=num_hiddens,
                                                num_residual_layers=num_residual_layers,
                                                num_residual_hiddens=num_residual_hiddens)

            self.encoder = nn.Sequential(
                self.conv_1,
                self.res_1,
                nn.ReLU(),
                self.conv_2,
                self.res_2,
                nn.ReLU(),
                self.conv_3,
                self.residual_stack
            )

        elif stride == 2:
            self.conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                    out_channels=num_hiddens,
                                    kernel_size=3,
                                    stride=1, padding=1)
            nn.init.kaiming_uniform_(self.conv_2.weight, nonlinearity='relu')
            
            self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                                num_hiddens=num_hiddens,
                                                num_residual_layers=num_residual_layers,
                                                num_residual_hiddens=num_residual_hiddens)

            self.encoder = nn.Sequential(
                self.conv_1,
                self.res_1,
                nn.ReLU(),
                self.conv_2,
                self.residual_stack
            )

        elif stride == 8:
            self.conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                    out_channels=num_hiddens,
                                    kernel_size=4,
                                    stride=2, padding=1)
            nn.init.kaiming_uniform_(self.conv_2.weight, nonlinearity='relu')
            self.res_2 = ResidualBlock(num_hiddens, num_hiddens, num_hiddens//4)

            self.conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                    out_channels=num_hiddens,
                                    kernel_size=4,
                                    stride=2, padding=1)
            nn.init.kaiming_uniform_(self.conv_3.weight, nonlinearity='relu')

            self.res_3 = ResidualBlock(num_hiddens, num_hiddens, num_hiddens//4)
            self.conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                    out_channels=num_hiddens,
                                    kernel_size=3,
                                    stride=1, padding=1)
            nn.init.kaiming_uniform_(self.conv_4.weight, nonlinearity='relu')
            
            self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                                num_hiddens=num_hiddens,
                                                num_residual_layers=num_residual_layers,
                                                num_residual_hiddens=num_residual_hiddens)

            self.encoder = nn.Sequential(
                self.conv_1,
                self.res_1,
                nn.ReLU(),
                self.conv_2,
                self.res_2,
                nn.ReLU(),
                self.conv_3,
                self.res_3,
                nn.ReLU(),
                self.conv_4,
                self.residual_stack
            )
        
        else:
            print(f"stride must be 2 or 4, not {stride}")
            return -1

    def forward(self, x):
        x = self.encoder(x)
        return x
    
    
    
class Decoder(nn.Module): # add out_channels
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens, stride):
        super(Decoder, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                out_channels=num_hiddens,
                                kernel_size=3, 
                                stride=1, padding=1)
        nn.init.kaiming_uniform_(self.conv_1.weight, nonlinearity='relu')
        
        self.residual_stack = ResidualStack(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=num_residual_hiddens)
                                            
        if stride == 4:
            self.conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens//2,
                                                    kernel_size=4, 
                                                    stride=2, padding=1)
            nn.init.kaiming_uniform_(self.conv_trans_1.weight, nonlinearity='relu')

            self.conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                    out_channels=out_channels,
                                                    kernel_size=4, 
                                                    stride=2, padding=1)
            nn.init.kaiming_uniform_(self.conv_trans_2.weight, nonlinearity='relu')

            self.decoder = nn.Sequential(
                self.conv_1,
                self.residual_stack,
                self.conv_trans_1,
                nn.ReLU(),
                self.conv_trans_2
            )
        
        elif stride == 2:
            self.conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                    out_channels=out_channels,
                                                    kernel_size=4, 
                                                    stride=2, padding=1)
            nn.init.kaiming_uniform_(self.conv_trans_1.weight, nonlinearity='relu')

            self.decoder = nn.Sequential(
                self.conv_1,
                self.residual_stack,
                self.conv_trans_1,
            )
        
        elif stride == 8:
            self.conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens,
                                                    kernel_size=4, 
                                                    stride=2, padding=1)
            nn.init.kaiming_uniform_(self.conv_trans_1.weight, nonlinearity='relu')

            self.conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                    out_channels=num_hiddens//2,
                                                    kernel_size=4, 
                                                    stride=2, padding=1)
            nn.init.kaiming_uniform_(self.conv_trans_2.weight, nonlinearity='relu')

            self.conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//2,
                                                    out_channels=out_channels,
                                                    kernel_size=4, 
                                                    stride=2, padding=1)
            nn.init.kaiming_uniform_(self.conv_trans_3.weight, nonlinearity='relu')

            self.decoder = nn.Sequential(
                self.conv_1,
                self.residual_stack,
                self.conv_trans_1,
                nn.ReLU(),
                self.conv_trans_2,
                nn.ReLU(),
                self.conv_trans_3
            )
        
        else:
            print(f"stride must be 2 or 4, not {stride}")
            return -1

    def forward(self, x):
        x = self.decoder(x)
        return x