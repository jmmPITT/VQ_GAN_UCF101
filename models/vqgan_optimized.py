import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# Helper modules
class ResidualBlock(nn.Module):
    """Optimized residual block with reduced parameters"""
    def __init__(self, in_channels, out_channels, use_norm=True, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main convolution layers with reduced parameters
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        # Use group norm with fewer groups for efficiency
        self.norm1 = nn.GroupNorm(min(4, out_channels), out_channels) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(min(4, out_channels), out_channels) if use_norm else nn.Identity()
        
        # Skip connection handling when channels don't match
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        
        # Activation and dropout
        self.activation = nn.SiLU(inplace=True)  # SiLU (Swish) for better performance
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        
        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        # Second conv block
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Skip connection
        out += identity
        out = self.activation(out)
        
        return out

class AttentionBlock(nn.Module):
    """Optimized self-attention block with reduced computation"""
    def __init__(self, channels, num_heads=2):  # Reduced heads from 4 to 2
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(min(4, channels), channels)
        # More efficient implementation using separate convs
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        
        # Normalize input
        x = self.norm(x)
        
        # Get query, key, value
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        
        # Compute attention weights with memory-efficient implementation
        attn = torch.einsum('bhci,bhcj->bhij', q, k) * (self.channels * self.num_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhij,bhcj->bhci', attn, v)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x=h, y=w)
        
        # Final projection
        out = self.proj(out)
        
        # Add residual connection
        return out + residual

class DownSampleBlock(nn.Module):
    """Optimized downsample block with optional attention"""
    def __init__(self, in_channels, out_channels, use_conv=True, use_attention=False):
        super().__init__()
        
        # Choose downsampling method
        if use_conv:
            # Downsampling with strided convolution (preferred for VQGAN)
            self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            # Downsampling with average pooling
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Residual block after downsampling
        self.res_block = ResidualBlock(in_channels, out_channels)
        
        # Optional attention block
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(out_channels)
    
    def forward(self, x):
        x = self.downsample(x)
        x = self.res_block(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        return x

class UpSampleBlock(nn.Module):
    """Optimized upsample block with optional attention"""
    def __init__(self, in_channels, out_channels, use_conv=True, use_attention=False):
        super().__init__()
        
        # Choose upsampling method - prefer interpolation over transposed convolution
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )
        
        # Residual block after upsampling
        self.res_block = ResidualBlock(in_channels, out_channels)
        
        # Optional attention block
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(out_channels)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.res_block(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        return x

class Encoder(nn.Module):
    """Optimized encoder with reduced complexity"""
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 256],  # Reduced channel dimensions
                 latent_dim=16, downsample_factor=4):
        super().__init__()
        
        # Initial input processing
        self.init_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1, bias=False)
        self.init_norm = nn.GroupNorm(min(4, hidden_dims[0]), hidden_dims[0])
        self.init_act = nn.SiLU(inplace=True)
        
        # Track sequence of layers for forward pass
        self.downsample_layers = nn.ModuleList()
        
        # Create downsampling layers
        current_resolution_level = 0  # Track how many times we've downsampled
        for i in range(len(hidden_dims) - 1):
            # Add attention only to the deeper layers and limit to save memory
            use_attention = (current_resolution_level >= (downsample_factor // 2)) and (i == len(hidden_dims) - 2)
            
            # Create downsampling block
            self.downsample_layers.append(
                DownSampleBlock(
                    hidden_dims[i], 
                    hidden_dims[i+1],
                    use_conv=True,  # Use convolution for downsampling
                    use_attention=use_attention
                )
            )
            
            current_resolution_level += 1
            
            # Add an additional residual block only at the middle layer
            if i == (len(hidden_dims) - 1) // 2:
                self.downsample_layers.append(
                    ResidualBlock(hidden_dims[i+1], hidden_dims[i+1])
                )
        
        # Final output convolution to get latent representation
        self.final_conv = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        # Initial input processing
        x = self.init_conv(x)
        x = self.init_norm(x)
        x = self.init_act(x)
        
        # Apply all downsampling layers
        for layer in self.downsample_layers:
            x = layer(x)
        
        # Final convolution to get latent space
        z = self.final_conv(x)
        
        return z

class Decoder(nn.Module):
    """Optimized decoder with reduced complexity"""
    def __init__(self, out_channels=3, hidden_dims=[256, 128, 64, 32],  # Reduced channel dimensions
                 latent_dim=16, upsample_factor=4):
        super().__init__()
        
        # Initial latent processing
        self.init_conv = nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=3, padding=1, bias=False)
        self.init_norm = nn.GroupNorm(min(4, hidden_dims[0]), hidden_dims[0])
        self.init_act = nn.SiLU(inplace=True)
        
        # Track sequence of layers for forward pass
        self.upsample_layers = nn.ModuleList()
        
        # Create upsampling layers
        current_resolution_level = upsample_factor  # Track resolution level (reversed from encoder)
        for i in range(len(hidden_dims) - 1):
            # Add attention only to the deeper layers and limit to save memory
            use_attention = (current_resolution_level >= (upsample_factor // 2)) and (i == 0)
            
            # Create upsampling block
            self.upsample_layers.append(
                UpSampleBlock(
                    hidden_dims[i], 
                    hidden_dims[i+1],
                    use_conv=False,  # Use interpolation for upsampling to save memory
                    use_attention=use_attention
                )
            )
            
            current_resolution_level -= 1
            
            # Add an additional residual block only at the middle layer
            if i == (len(hidden_dims) - 1) // 2:
                self.upsample_layers.append(
                    ResidualBlock(hidden_dims[i+1], hidden_dims[i+1])
                )
        
        # Final output convolution to get reconstructed image
        self.final_conv = nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1)
        self.final_act = nn.Sigmoid()  # Use sigmoid for [0,1] image output range
        
    def forward(self, x):
        # Initial latent processing
        x = self.init_conv(x)
        x = self.init_norm(x)
        x = self.init_act(x)
        
        # Apply all upsampling layers
        for layer in self.upsample_layers:
            x = layer(x)
        
        # Final convolution and activation to get reconstructed image
        x = self.final_conv(x)
        x = self.final_act(x)
        
        return x

class VectorQuantizer(nn.Module):
    """Optimized Vector Quantizer with improved efficiency"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.5, decay=0.99, 
                 eps=1e-5, restart_unused=True):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        self.restart_unused = restart_unused
        
        # Initialize embeddings - normalize for better training stability
        embed = torch.randn(num_embeddings, embedding_dim)
        embed = F.normalize(embed, p=2, dim=1)
        self.register_buffer('embeddings', embed)
        
        # EMA tracking buffers
        self.register_buffer('ema_count', torch.ones(num_embeddings))
        self.register_buffer('ema_weight', self.embeddings.clone())
        
        # Track codebook usage for visualization and analysis
        self.register_buffer('usage', torch.zeros(num_embeddings))
        
    def forward(self, z):
        # Save input shape for later
        input_shape = z.shape
        
        # Ensure embeddings have same dtype as input
        if self.embeddings.dtype != z.dtype:
            self.embeddings = self.embeddings.to(z.dtype)
            self.ema_weight = self.ema_weight.to(z.dtype)
            self.ema_count = self.ema_count.to(z.dtype)
            self.usage = self.usage.to(z.dtype)
        
        # Flatten input for easier processing
        flat_z = z.view(-1, self.embedding_dim)
        
        # Compute L2 distance in a memory-efficient way
        z_norm = torch.sum(flat_z**2, dim=1, keepdim=True)
        e_norm = torch.sum(self.embeddings**2, dim=1)
        
        # Calculate distances
        dist = z_norm + e_norm - 2 * torch.matmul(flat_z, self.embeddings.t())
        
        # Find nearest embedding for each input vector
        encoding_indices = torch.argmin(dist, dim=1)
        
        # More memory-efficient one-hot encoding - use same dtype as input
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, 
                              device=z.device, dtype=flat_z.dtype)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Get quantized vectors from the codebook
        quantized = torch.matmul(encodings, self.embeddings)
        
        # Reshape to original dimensions
        quantized = quantized.view(input_shape)
        
        # Update usage stats during training
        if self.training:
            with torch.no_grad():
                # Update less frequently to save compute (every other forward pass)
                if torch.rand(1).item() < 0.5:
                    self.usage = self.decay * self.usage + (1 - self.decay) * encodings.sum(0)
                    
                    # EMA codebook update
                    encodings_sum = encodings.sum(0)
                    self.ema_count = self.decay * self.ema_count + (1 - self.decay) * encodings_sum
                    
                    # Ensure counts don't get too small
                    n = torch.sum(self.ema_count)
                    self.ema_count = (self.ema_count + self.eps) / (n + self.num_embeddings * self.eps) * n
                    
                    # Update embeddings with EMA
                    dw = torch.matmul(encodings.t(), flat_z)
                    self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
                    
                    # Normalize and update embeddings
                    self.embeddings = self.ema_weight / self.ema_count.unsqueeze(1)
                
                # Restart unused embeddings less frequently to save compute
                if self.restart_unused and torch.rand(1).item() < 0.005:  # 0.5% chance per batch
                    # Find unused embeddings (using a dynamic threshold)
                    usage_thresh = torch.quantile(self.usage, 0.05)  # Bottom 5%
                    unused_indices = torch.where(self.usage < usage_thresh)[0]
                    
                    if len(unused_indices) > 0:
                        # Choose random samples from the batch
                        n_to_restart = min(len(unused_indices), 4)  # Limit number of restarts per batch
                        selected_unused = unused_indices[torch.randperm(len(unused_indices))[:n_to_restart]]
                        
                        # Choose random encoded vectors from the current batch
                        sampled_indices = torch.randint(0, flat_z.shape[0], (n_to_restart,), device=z.device)
                        sampled_vectors = flat_z[sampled_indices]
                        
                        # Add small random noise for diversity
                        noise = torch.randn_like(sampled_vectors) * 0.1
                        new_embeddings = sampled_vectors + noise
                        
                        # Normalize the new embeddings
                        new_embeddings = F.normalize(new_embeddings, p=2, dim=1)
                        
                        # Update the selected embeddings
                        self.embeddings[selected_unused] = new_embeddings
                        self.ema_weight[selected_unused] = new_embeddings * self.ema_count[selected_unused].unsqueeze(1)
                
        # Calculate losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        
        # Use commitment cost to balance the two losses
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = z + (quantized - z).detach()
        
        # Reshape indices for returning
        encoding_indices = encoding_indices.view(input_shape[0], *input_shape[2:])
        
        return quantized, vq_loss, encoding_indices
    
    def get_codebook_usage(self):
        """Return the current usage counts for visualization"""
        return self.usage

class Discriminator(nn.Module):
    """Optimized PatchGAN discriminator with fewer parameters"""
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 256], 
                 patch_size=70):
        super().__init__()
        
        layers = []
        
        # Initial convolution
        layers.append(nn.Conv2d(in_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers with increasing channels and downsampling
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i+1]
            
            # Add convolutional layer with spectral norm for stability
            # Use fewer channels and remove bias
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False)
            ))
            
            # Use LeakyReLU for discriminator
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layer to produce 1-channel output (predictions)
        layers.append(nn.Conv2d(hidden_dims[-1], 1, kernel_size=1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class OptimizedVQGAN(nn.Module):
    """Memory-efficient VQGAN model with encoder, quantizer, and decoder"""
    def __init__(
        self, 
        in_channels=3, 
        hidden_dims=[32, 64, 128, 256],  # Reduced channel dimensions
        latent_dim=16,  # Reduced latent dimension
        num_embeddings=256,  # Reduced codebook size 
        embedding_dim=16,
        downsample_factor=4,
        commitment_cost=0.25,
        decay=0.99,
        restart_unused=True
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Ensure latent dimension matches embedding dimension
        assert latent_dim == embedding_dim, "Latent dimension must match embedding dimension"
        
        # Create encoder, vector quantizer, and decoder
        self.encoder = Encoder(
            in_channels=in_channels, 
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            downsample_factor=downsample_factor
        )
        
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            restart_unused=restart_unused
        )
        
        self.decoder = Decoder(
            out_channels=in_channels,
            hidden_dims=hidden_dims[::-1],  # Reverse hidden dims for decoder
            latent_dim=latent_dim,
            upsample_factor=downsample_factor
        )
        
    def encode(self, x):
        """Encode and quantize the input"""
        z = self.encoder(x)
        quantized, vq_loss, indices = self.vector_quantizer(z)
        return z, quantized, vq_loss, indices
    
    def decode(self, quantized):
        """Decode the quantized representation"""
        return self.decoder(quantized)
    
    def forward(self, x, return_perceptual=False, return_indices=False):
        """Forward pass through the VQ-GAN model"""
        # Encode and quantize
        z, quantized, vq_loss, indices = self.encode(x)
        
        # Decode
        reconstructions = self.decode(quantized)
        
        # Return appropriate values based on what was requested
        if return_indices and return_perceptual:
            return reconstructions, quantized, vq_loss, z, indices
        elif return_indices:
            return reconstructions, quantized, vq_loss, indices
        elif return_perceptual:
            return reconstructions, quantized, vq_loss, z
        else:
            return reconstructions, quantized, vq_loss
    
    def get_codebook_usage(self):
        """Return the current codebook usage"""
        return self.vector_quantizer.get_codebook_usage()