use burn::backend::{Autodiff, Wgpu};
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig, PaddingConfig2d};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::ElementConversion;
use burn::tensor::Int;
use burn::tensor::Tensor;
use image::{DynamicImage, ImageBuffer, Rgb};
use std::path::PathBuf;

// Add egobox imports for Bayesian Optimization
use egobox_doe::{Lhs, SamplingMethod};
use egobox_ego::{EgorBuilder, InfillStrategy, InfillOptimizer};
use egobox_gp::{GpMixture, GpMixtureParams};
use ndarray::{Array1, Array2, array};

#[derive(Module, Debug)]
pub struct SimpleEncoder<B: Backend> {
    conv1: Conv2d<B>,
    bn_1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn_2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    bn_3: BatchNorm<B, 2>,
    mean: Linear<B>,
    variance: Linear<B>,
    latent_dimen: usize,
}

impl<B: Backend> SimpleEncoder<B> {
    pub fn new(in_channels: usize, latent_dimen: usize, device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([in_channels, 32], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            bn_1: BatchNormConfig::new(32).init(device),

            conv2: Conv2dConfig::new([32, 64], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            bn_2: BatchNormConfig::new(64).init(device),

            conv3: Conv2dConfig::new([64, 128], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            bn_3: BatchNormConfig::new(128).init(device),

            mean: LinearConfig::new(128 * 16 * 16, latent_dimen * 16).init(device),
            variance: LinearConfig::new(128 * 16 * 16, latent_dimen * 16).init(device),
            latent_dimen,
        }
    }

    pub fn forward(&self, input1: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = self.bn_1.forward(self.conv1.forward(input1));
        let x = activation::relu(x);

        let x = self.bn_2.forward(self.conv2.forward(x));
        let x = activation::relu(x);

        let x = self.bn_3.forward(self.conv3.forward(x));
        let x = activation::relu(x);

        let x = x.flatten(1, 3);

        let mean = self.mean.forward(x.clone());
        let variance = self.variance.forward(x.clone());

        (mean, variance)
    }
}

#[derive(Module, Debug)]
pub struct SimpleDecoder<B: Backend> {
    transfer: Linear<B>,
    reverse1: ConvTranspose2d<B>,
    bn_1: BatchNorm<B, 2>,
    reverse2: ConvTranspose2d<B>,
    bn_2: BatchNorm<B, 2>,
    reverse3: ConvTranspose2d<B>,
    latent_dimen: usize,
    output: usize,
}

impl<B: Backend> SimpleDecoder<B> {
    pub fn new(output: usize, latent_dimen: usize, device: &B::Device) -> Self {
        Self {
            transfer: LinearConfig::new(latent_dimen * 16, 128 * 16 * 16).init(device),

            reverse1: ConvTranspose2dConfig::new([128, 64], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            bn_1: BatchNormConfig::new(64).init(device),

            reverse2: ConvTranspose2dConfig::new([64, 32], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            bn_2: BatchNormConfig::new(32).init(device),

            reverse3: ConvTranspose2dConfig::new([32, output], [3, 3])
                .with_stride([1, 1])
                .with_padding([1, 1])
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),

            latent_dimen,
            output,
        }
    }

    pub fn forward(&self, input2: Tensor<B, 4>) -> Tensor<B, 4> {
        let input2_flat: Tensor<B, 2> = input2.flatten(1, 3);
        let y = self.transfer.forward(input2_flat);
        let y = activation::relu(y);
        let batch_size = y.dims()[0];
        let y = y.reshape([batch_size, 128, 16, 16]);

        let y = self.reverse1.forward(y);
        let y = self.bn_1.forward(y);
        let y = activation::relu(y);

        let y = self.reverse2.forward(y);
        let y = self.bn_2.forward(y);
        let y = activation::relu(y);

        let y = self.reverse3.forward(y);
        activation::tanh(y)
    }
}

#[derive(Module, Debug)]
pub struct Vae<B: Backend> {
    encoder: SimpleEncoder<B>,
    decoder: SimpleDecoder<B>,
}

impl<B: Backend> Vae<B> {
    pub fn new(in_channels: usize, latent_dimen: usize, device: &B::Device) -> Self {
        Self {
            encoder: SimpleEncoder::new(in_channels, latent_dimen, device),
            decoder: SimpleDecoder::new(in_channels, latent_dimen, device),
        }
    }

    pub fn parametrize(&self, mean: Tensor<B, 2>, variance: Tensor<B, 2>) -> Tensor<B, 4> {
        let variance = variance.clamp(-10.0, 10.0);
        let std = (variance.clone() * 0.5).exp();
        let noise = Tensor::random_like(&std, burn::tensor::Distribution::Normal(0.0, 1.0));
        let z = mean + noise * std;

        z.clone()
            .reshape([z.dims()[0], self.encoder.latent_dimen, 4, 4])
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>) {
        let (mean, variance) = self.encoder.forward(x);
        let z = self.parametrize(mean.clone(), variance.clone());
        let reconstruction = self.decoder.forward(z);
        (reconstruction, mean, variance)
    }

    pub fn encode(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (mean, variance) = self.encoder.forward(x);
        self.parametrize(mean, variance)
    }

    pub fn decode(&self, y: Tensor<B, 4>) -> Tensor<B, 4> {
        self.decoder.forward(y)
    }

    pub fn decode_from_mean(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (mean, _variance) = self.encoder.forward(x);
        let z = mean.reshape([mean.dims()[0], self.encoder.latent_dimen, 4, 4]);
        self.decoder.forward(z)
    }

    pub fn vae_loss(
        mean: Tensor<B, 2>,
        variance: Tensor<B, 2>,
        reconstruction: Tensor<B, 4>,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let epsilon = 1e-8;
        let variance = variance + epsilon;
        let kl_loss = -0.5
            * (Tensor::ones_like(&variance) + variance.clone()
                - mean.powf_scalar(2.0)
                - variance.exp())
            .mean();
        let recon_flat: Tensor<B, 2> = reconstruction.flatten(1, 3);
        let input_flat: Tensor<B, 2> = input.flatten(1, 3);
        let recon_loss = burn::nn::loss::MseLoss::new().forward(
            recon_flat,
            input_flat,
            burn::nn::loss::Reduction::Mean,
        );
        recon_loss + kl_loss
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    question: Linear<B>,
    key: Linear<B>,
    val: Linear<B>,
    out: Linear<B>,
    channels: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            question: LinearConfig::new(channels, channels).init(device),
            key: LinearConfig::new(channels, channels).init(device),
            val: LinearConfig::new(channels, channels).init(device),
            out: LinearConfig::new(channels, channels).init(device),
            channels,
        }
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [a, b, c, d] = x.dims();
        let x_flat = x.clone().reshape([a, b, c * d]);
        let x_flated = x_flat.swap_dims(1, 2);

        let q = self.question.forward(x_flated.clone());
        let k = self.key.forward(x_flated.clone());
        let v = self.val.forward(x_flated.clone());

        let scaling = (self.channels as f32).sqrt();
        let k_transpose = k.swap_dims(1, 2);
        let score = q.matmul(k_transpose) / scaling;
        let attn_weights = activation::softmax(score, 2);

        let attn_output = attn_weights.matmul(v);
        let attn_output = self.out.forward(attn_output);

        let attn_output = attn_output.swap_dims(1, 2).reshape([a, b, c, d]);

        attn_output + x
    }
}

#[derive(Module, Debug)]
pub struct TimeAddition<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> TimeAddition<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(dim, dim * 8)
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            linear2: LinearConfig::new(dim * 8, dim)
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
        }
    }

    pub fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
        let z = self.linear1.forward(t);
        let z = activation::relu(z);
        self.linear2.forward(z)
    }
}

pub fn sin_time_addition<B: Backend>(
    device: &B::Device,
    time: Tensor<B, 1, Int>,
    dim: usize,
) -> Tensor<B, 2> {
    let batch_size = time.dims()[0];
    let half_dim = dim / 2;
    let t = time.float();

    let frequencies: Vec<f32> = (0..half_dim)
        .map(|i| {
            let exp = (i as f32) * 4.0 * std::f32::consts::LN_10 / (half_dim as f32);
            (-exp).exp()
        })
        .collect();

    let freqs: Tensor<B, 2> = Tensor::<B, 1>::from_floats(frequencies.as_slice(), device)
        .reshape([1, half_dim])
        .repeat_dim(0, batch_size);

    let arg = t.reshape([batch_size, 1]).repeat_dim(1, half_dim) * freqs;

    let sin_emb = arg.clone().sin();
    let cos_emb = arg.cos();

    Tensor::cat(vec![sin_emb, cos_emb], 1)
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    bn2: BatchNorm<B, 2>,
    time: Linear<B>,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn new(device: &B::Device, channels: usize, time_emb_dim: usize) -> Self {
        Self {
            conv1: Conv2dConfig::new([channels, channels], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            bn1: BatchNormConfig::new(channels).init(device),

            conv2: Conv2dConfig::new([channels, channels], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            bn2: BatchNormConfig::new(channels).init(device),

            time: LinearConfig::new(time_emb_dim, channels).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, t_emb: Tensor<B, 2>) -> Tensor<B, 4> {
        let h = self.conv1.forward(x.clone());
        let h = self.bn1.forward(h);
        let h = activation::silu(h);

        let t = self.time.forward(t_emb);
        let t = activation::silu(t);
        let batch_size = t.dims()[0];
        let channels = t.dims()[1];
        let t = t.reshape([batch_size, channels, 1, 1]);
        let h = h + t;

        let h = self.conv2.forward(h);
        let h = self.bn2.forward(h);
        activation::silu(h + x)
    }
}

#[derive(Module, Debug)]
pub struct Unet<B: Backend> {
    down1: Conv2d<B>,
    down_res1: ResidualBlock<B>,
    down2: Conv2d<B>,
    down_res2: ResidualBlock<B>,
    down3: Conv2d<B>,
    down_res3: ResidualBlock<B>,

    mid_res1: ResidualBlock<B>,
    mid_res2: ResidualBlock<B>,
    mid_atten: Attention<B>,

    up1: ConvTranspose2d<B>,
    up_conv1: Conv2d<B>,
    up_res1: ResidualBlock<B>,

    up2: ConvTranspose2d<B>,
    up_conv2: Conv2d<B>,
    up_res2: ResidualBlock<B>,

    out_conv: Conv2d<B>,
    time_emb: TimeAddition<B>,
}

impl<B: Backend> Unet<B> {
    pub fn new(device: &B::Device, in_channels: usize, time_emb_dim: usize) -> Self {
        Self {
            time_emb: TimeAddition::new(time_emb_dim, device),

            down1: Conv2dConfig::new([in_channels, 64], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            down_res1: ResidualBlock::new(device, 64, time_emb_dim),

            down2: Conv2dConfig::new([64, 128], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            down_res2: ResidualBlock::new(device, 128, time_emb_dim),

            down3: Conv2dConfig::new([128, 256], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            down_res3: ResidualBlock::new(device, 256, time_emb_dim),

            mid_res1: ResidualBlock::new(device, 256, time_emb_dim),
            mid_res2: ResidualBlock::new(device, 256, time_emb_dim),
            mid_atten: Attention::new(256, device),

            up1: ConvTranspose2dConfig::new([256, 128], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            up_conv1: Conv2dConfig::new([256, 128], [1, 1])
                .with_stride([1, 1])
                .init(device),
            up_res1: ResidualBlock::new(device, 128, time_emb_dim),

            up2: ConvTranspose2dConfig::new([128, 64], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
            up_conv2: Conv2dConfig::new([128, 64], [1, 1])
                .with_stride([1, 1])
                .init(device),
            up_res2: ResidualBlock::new(device, 64, time_emb_dim),

            out_conv: Conv2dConfig::new([64, in_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_initializer(Initializer::KaimingUniform {
                    gain: (0.1),
                    fan_out_only: (false),
                })
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, time: Tensor<B, 2>) -> Tensor<B, 4> {
        let time_embed = self.time_emb.forward(time);

        let down_1 = self.down1.forward(x);
        let down_1 = self.down_res1.forward(down_1, time_embed.clone());

        let down_2 = self.down2.forward(down_1.clone());
        let down_2 = self.down_res2.forward(down_2, time_embed.clone());

        let down_3 = self.down3.forward(down_2.clone());
        let down_3 = self.down_res3.forward(down_3, time_embed.clone());

        let mut m = self.mid_res1.forward(down_3, time_embed.clone());
        m = self.mid_atten.forward(m);
        m = self.mid_res2.forward(m, time_embed.clone());

        let u1 = self.up1.forward(m);
        let u1 = Tensor::cat(vec![u1, down_2.clone()], 1);
        let u1 = self.up_conv1.forward(u1);
        let u1 = self.up_res1.forward(u1, time_embed.clone());

        let u2 = self.up2.forward(u1);
        let u2 = Tensor::cat(vec![u2, down_1.clone()], 1);
        let u2 = self.up_conv2.forward(u2);
        let u2 = self.up_res2.forward(u2, time_embed.clone());

        self.out_conv.forward(u2)
    }
}

#[derive(Module, Debug)]
pub struct DiffusionModel<B: Backend> {
    unet: Unet<B>,
    vae: Vae<B>,
    num_time: usize,
    latent_dimen: usize,
}

impl<B: Backend> DiffusionModel<B> {
    pub fn new(
        device: &B::Device,
        latent_dimen: usize,
        in_channels: usize,
        num_time: usize,
    ) -> Self {
        Self {
            vae: Vae::new(in_channels, latent_dimen, device),
            unet: Unet::new(device, latent_dimen, 256),
            num_time,
            latent_dimen,
        }
    }

    pub fn get_betas(&self, device: &B::Device) -> Tensor<B, 1> {
        let beta_start = 0.0001;
        let beta_end = 0.02;

        let steps = self.num_time;
        let betas: Vec<f32> = (0..steps)
            .map(|i| beta_start + (beta_end - beta_start) * (i as f32) / ((steps - 1) as f32))
            .collect();

        Tensor::from_floats(betas.as_slice(), device)
    }

    pub fn get_alphas(&self, device: &B::Device) -> Tensor<B, 1> {
        let betas = self.get_betas(device);
        let alphas = Tensor::ones_like(&betas) - betas;
        let alpha_vals: Vec<f32> = alphas.to_data().to_vec().unwrap();
        let mut cumulative = 1.0;
        let alpha_bars: Vec<f32> = alpha_vals
            .iter()
            .map(|&a| {
                cumulative *= a;
                cumulative
            })
            .collect();

        Tensor::from_floats(alpha_bars.as_slice(), device)
    }

    pub fn add_noise(
        &self,
        x: Tensor<B, 4>,
        time: Tensor<B, 1, Int>,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let alpha = self.get_alphas(device);
        let noise = Tensor::random_like(&x, burn::tensor::Distribution::Normal(0.0, 1.0));

        let batch_size = x.dims()[0];
        let channels = x.dims()[1];
        let height = x.dims()[2];
        let width = x.dims()[3];

        let time_values: Vec<i32> = time.to_data().to_vec().unwrap();
        let alpha_values: Vec<f32> = alpha.to_data().to_vec().unwrap();

        let gathered_alphas: Vec<f32> = time_values
            .iter()
            .map(|&t| alpha_values[t as usize])
            .collect();

        let alpha_t: Tensor<B, 1> = Tensor::from_floats(gathered_alphas.as_slice(), device);
        let alpha_t = alpha_t
            .reshape([batch_size, 1, 1, 1])
            .repeat_dim(1, channels)
            .repeat_dim(2, height)
            .repeat_dim(3, width);

        let sqrt_alpha = alpha_t.clone().sqrt();
        let sqrt_one_minus_alpha = (Tensor::ones_like(&alpha_t) - alpha_t).sqrt();

        let noisy = x * sqrt_alpha + noise.clone() * sqrt_one_minus_alpha;

        (noisy, noise)
    }

    pub fn noise_predict(&self, x_t: Tensor<B, 4>, time_emb: Tensor<B, 2>) -> Tensor<B, 4> {
        self.unet.forward(x_t, time_emb)
    }

    pub fn forward(&self, x: Tensor<B, 4>, device: &B::Device) -> Tensor<B, 1> {
        let batch_size = x.dims()[0];
        let (mean, variance) = self.vae.encoder.forward(x.clone());

        let z = self.vae.parametrize(mean.clone(), variance.clone());

        let reconstruction = self.vae.decoder.forward(z.clone());
        let total_vae_loss = Vae::vae_loss(mean, variance, reconstruction, x);

        let t_values: Vec<i32> = (0..batch_size)
            .map(|_| (rand::random::<u32>() % self.num_time as u32) as i32)
            .collect();

        let t = Tensor::from_ints(t_values.as_slice(), device);
        let (z_noisy, noise) = self.add_noise(z, t.clone(), device);

        let t_emb = sin_time_addition(device, t, 256);

        let noise_pred = self.noise_predict(z_noisy, t_emb);
        let noise_pred_2d: Tensor<B, 2> = noise_pred.flatten(1, 3);

        let noise_2d: Tensor<B, 2> = noise.flatten(1, 3);
        let diffusion_loss = (noise_pred_2d - noise_2d).powf_scalar(2.0).mean();

        let diffusion_loss_1d = diffusion_loss.reshape([1]);

        total_vae_loss + diffusion_loss_1d
    }

    // CORRECTED SAMPLING FUNCTION - Proper DDPM Algorithm
    pub fn sample(&self, batch_size: usize, device: &B::Device) -> Tensor<B, 4> {
        // Start from pure noise
        let mut z: Tensor<B, 4> = Tensor::random(
            [batch_size, self.latent_dimen, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );

        let betas = self.get_betas(device);
        let alphas_bar = self.get_alphas(device); // These are cumulative products ᾱ_t
        let betas_vec: Vec<f32> = betas.to_data().to_vec().unwrap();
        let alphas_bar_vec: Vec<f32> = alphas_bar.to_data().to_vec().unwrap();
        
        // Compute α_t (single step, not cumulative)
        let alphas_vec: Vec<f32> = betas_vec.iter().map(|&b| 1.0 - b).collect();

        println!("\n🎨 Starting reverse diffusion sampling...");
        
        // Reverse diffusion process: from T to 0
        for i in (0..self.num_time).rev() {
            let i_tensor = Tensor::from_ints(vec![i as i32; batch_size].as_slice(), device);
            let i_emb = sin_time_addition(device, i_tensor, 256);

            // Predict noise using U-Net
            let noise_pred = self.noise_predict(z.clone(), i_emb);

            // Get current timestep parameters
            let alpha_t = alphas_vec[i];
            let alpha_bar_t = alphas_bar_vec[i];
            let beta_t = betas_vec[i];

            let [b, c, h, w] = z.dims();

            if i > 0 {
                // Standard DDPM update for t > 0
                let alpha_bar_prev = alphas_bar_vec[i - 1];
                
                // Compute posterior variance: β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
                let posterior_variance = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * beta_t;
                let posterior_variance_clipped = posterior_variance.max(1e-20);
                
                // Predict x_0 from x_t and noise: x̂_0 = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t
                let sqrt_alpha_bar_t = alpha_bar_t.sqrt();
                let sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt();
                
                let sqrt_alpha_bar_t_tensor = Tensor::from_floats(&[sqrt_alpha_bar_t], device)
                    .reshape([1, 1, 1, 1])
                    .expand([b, c, h, w]);
                let sqrt_one_minus_alpha_bar_t_tensor = Tensor::from_floats(&[sqrt_one_minus_alpha_bar_t], device)
                    .reshape([1, 1, 1, 1])
                    .expand([b, c, h, w]);
                
                let pred_x0 = (z.clone() - noise_pred.clone() * sqrt_one_minus_alpha_bar_t_tensor) 
                    / sqrt_alpha_bar_t_tensor;
                
                // Compute mean of posterior: μ̃_t = √ᾱ_{t-1}·β_t/(1-ᾱ_t)·x̂_0 + √α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)·x_t
                let coef1 = (alpha_bar_prev.sqrt() * beta_t) / (1.0 - alpha_bar_t);
                let coef2 = (alpha_t.sqrt() * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar_t);
                
                let coef1_tensor = Tensor::from_floats(&[coef1], device)
                    .reshape([1, 1, 1, 1])
                    .expand([b, c, h, w]);
                let coef2_tensor = Tensor::from_floats(&[coef2], device)
                    .reshape([1, 1, 1, 1])
                    .expand([b, c, h, w]);
                
                let mean = pred_x0 * coef1_tensor + z.clone() * coef2_tensor;
                
                // Add noise: z_{t-1} = μ̃_t + √β̃_t·ε where ε ~ N(0, I)
                let noise = Tensor::random_like(&z, burn::tensor::Distribution::Normal(0.0, 1.0));
                let sigma = posterior_variance_clipped.sqrt();
                let sigma_tensor = Tensor::from_floats(&[sigma], device)
                    .reshape([1, 1, 1, 1])
                    .expand([b, c, h, w]);
                
                z = mean + sigma_tensor * noise;
                
            } else {
                // Final step (t=0): just predict x_0 without adding noise
                let sqrt_alpha_bar_t = alpha_bar_t.sqrt();
                let sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt();
                
                let sqrt_alpha_bar_t_tensor = Tensor::from_floats(&[sqrt_alpha_bar_t], device)
                    .reshape([1, 1, 1, 1])
                    .expand([b, c, h, w]);
                let sqrt_one_minus_alpha_bar_t_tensor = Tensor::from_floats(&[sqrt_one_minus_alpha_bar_t], device)
                    .reshape([1, 1, 1, 1])
                    .expand([b, c, h, w]);
                
                z = (z.clone() - noise_pred * sqrt_one_minus_alpha_bar_t_tensor) 
                    / sqrt_alpha_bar_t_tensor;
            }

            // Progress logging
            if i % 50 == 0 || i == 0 {
                println!("  Denoising step {}/{} (t={})", self.num_time - i, self.num_time, i);
            }
        }
        
        println!("✅ Reverse diffusion complete, decoding latent...");
        
        // Decode the denoised latent representation
        self.vae.decode(z)
    }

    pub fn train_vae(&self, x: Tensor<B, 4>, kl_weight: f32) -> Tensor<B, 1> {
        let (reconstruction, mean, variance) = self.vae.forward(x.clone());

        let epsilon = 1e-8;
        let variance_safe = variance.clone() + epsilon;

        let variance_crt = variance_safe.clamp(-10.0, 10.0);

        let kl_loss: Tensor<B, 1> = -0.5
            * (Tensor::ones_like(&variance_crt) + variance_crt.clone()
                - mean.powf_scalar(2.0)
                - variance_crt.exp())
            .mean();

        let recon_flat: Tensor<B, 2> = reconstruction.flatten(1, 3);
        let input_flat: Tensor<B, 2> = x.flatten(1, 3);
        let recon_loss = burn::nn::loss::MseLoss::new().forward(
            recon_flat,
            input_flat,
            burn::nn::loss::Reduction::Mean,
        );

        let kl_loss_1d: Tensor<B, 1> = kl_loss.mul_scalar(kl_weight).reshape([1]);
        recon_loss + kl_loss_1d
    }
}

pub fn save_model<B: Backend>(
    model: &DiffusionModel<B>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let recorder = CompactRecorder::new();
    recorder.record(model.clone().into_record(), path.into())?;
    println!("Model saved to {}", path);
    Ok(())
}

pub fn load_model<B: Backend>(
    path: &str,
    in_channels: usize,
    latent_dimen: usize,
    num_time: usize,
    device: &B::Device,
) -> Result<DiffusionModel<B>, Box<dyn std::error::Error>> {
    let recorder = CompactRecorder::new();
    let record = recorder.load(path.into(), device)?;

    let model =
        DiffusionModel::new(device, latent_dimen, in_channels, num_time).load_record(record);

    Ok(model)
}

pub fn save_tensor_as_image<B: Backend>(
    tensor: Tensor<B, 4>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let img_tensor: Tensor<B, 3> = tensor.squeeze::<3>(0);
    let [_channels, height, width] = img_tensor.dims();
    let data: Vec<f32> = img_tensor.to_data().to_vec().unwrap();

    let img = image::RgbImage::from_fn(width as u32, height as u32, |x, y| {
        let pixel_idx = (y as usize * width) + x as usize;
        let temp_r: f32 = data[pixel_idx] * 0.5 + 0.5;
        let clamped_r = temp_r.clamp(0.0, 1.0);
        let r = (clamped_r * 255.0) as u8;
        let temp_g: f32 = data[width * height + pixel_idx] * 0.5 + 0.5;
        let clamped_g = temp_g.clamp(0.0, 1.0);
        let g = (clamped_g * 255.0) as u8;
        let temp_b: f32 = data[2 * width * height + pixel_idx] * 0.5 + 0.5;
        let clamped_b = temp_b.clamp(0.0, 1.0);
        let b = (clamped_b * 255.0) as u8;
        image::Rgb([r, g, b])
    });

    img.save(path)?;
    Ok(())
}

#[derive(Module, Debug, Clone)]
pub struct Augmentation {
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
    pub rotation: bool,
    pub brightness: Option<f32>,
    pub contrast: Option<f32>,
}

impl Augmentation {
    pub fn new() -> Self {
        Self {
            horizontal_flip: true,
            vertical_flip: false,
            rotation: true,
            brightness: Some(0.2),
            contrast: Some(0.2),
        }
    }

    pub fn with_horizontal(mut self, flipped: bool) -> Self {
        self.horizontal_flip = flipped;
        self
    }

    pub fn with_vertical(mut self, flipped: bool) -> Self {
        self.vertical_flip = flipped;
        self
    }

    pub fn with_rotation(mut self, flipped: bool) -> Self {
        self.rotation = flipped;
        self
    }

    pub fn with_brightness(mut self, range: f32) -> Self {
        self.brightness = Some(range);
        self
    }

    pub fn with_contrast(mut self, range: f32) -> Self {
        self.contrast = Some(range);
        self
    }

    pub fn apply(&self, img: DynamicImage) -> DynamicImage {
        let mut img = img;
        if self.horizontal_flip && rand::random::<f32>() > 0.5 {
            img = img.fliph();
        }

        if self.vertical_flip && rand::random::<f32>() > 0.5 {
            img = img.flipv();
        }

        if self.rotation && rand::random::<f32>() > 0.8 {
            let rotation = rand::random::<u32>() % 3 + 1;
            img = img.rotate90();
            if rotation >= 3 {
                img = img.rotate90();
            }
        }

        if let Some(brightness_range) = self.brightness {
            let factor = 1.0 + (rand::random::<f32>() * 2.0 - 1.0) * brightness_range;
            img = adjust_brightness(img, factor);
        }
        if let Some(contrast_range) = self.contrast {
            let factor = 1.0 * (rand::random::<f32>() * 2.0 - 1.0) * contrast_range;
            img = adjust_contrast(img, factor);
        }
        img
    }
}

fn adjust_brightness(img: DynamicImage, factor: f32) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let adjusted = ImageBuffer::from_fn(width, height, |x, y| {
        let pixel = rgb.get_pixel(x, y);
        Rgb([
            (pixel[0] as f32 * factor).clamp(0.0, 255.0) as u8,
            (pixel[1] as f32 * factor).clamp(0.0, 255.0) as u8,
            (pixel[2] as f32 * factor).clamp(0.0, 255.0) as u8,
        ])
    });

    DynamicImage::ImageRgb8(adjusted)
}

fn adjust_contrast(img: DynamicImage, factor: f32) -> DynamicImage {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut sum = 0.0f32;
    for pixel in rgb.pixels() {
        sum += (pixel[0] as f32 + pixel[1] as f32 + pixel[2] as f32) / 3.0;
    }
    let mean = sum / (width * height) as f32;

    let adjusted = ImageBuffer::from_fn(width, height, |x, y| {
        let pixel = rgb.get_pixel(x, y);
        Rgb([
            (mean + (pixel[0] as f32 - mean) * factor).clamp(0.0, 255.0) as u8,
            (mean + (pixel[1] as f32 - mean) * factor).clamp(0.0, 255.0) as u8,
            (mean + (pixel[2] as f32 - mean) * factor).clamp(0.0, 255.0) as u8,
        ])
    });

    DynamicImage::ImageRgb8(adjusted)
}

pub struct Image {
    pub size: usize,
    pub channels: usize,
    pub image_paths: Vec<PathBuf>,
    pub height: usize,
    pub width: usize,
    pub augmentation: Option<Augmentation>,
}

impl Image {
    pub fn directory(
        dir_path: &str,
        height: usize,
        width: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use std::fs;
        use std::path::Path;

        let path = Path::new(dir_path);

        if !path.exists() {
            return Err(format!("Directory does not exist: {}", dir_path).into());
        }
        let mut image_paths = Vec::new();

        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(ext) = path.extension() {
                let ext = ext.to_str().unwrap_or("").to_lowercase();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif") {
                    image_paths.push(path);
                }
            }
        }

        if image_paths.is_empty() {
            return Err(format!("No images found in directory: {}", dir_path).into());
        }

        image_paths.sort();
        let size = image_paths.len();

        println!("Loaded {} images from {}", size, dir_path);

        Ok(Self {
            image_paths,
            size,
            channels: 3,
            height,
            width,
            augmentation: None,
        })
    }

    pub fn from_paths(
        paths: Vec<String>,
        height: usize,
        width: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut image_paths = Vec::new();

        for path_str in paths {
            let path = PathBuf::from(path_str);
            if !path.exists() {
                println!("Warning: File does not exist: {:?}", path);
                continue;
            }
            image_paths.push(path);
        }

        if image_paths.is_empty() {
            return Err("No valid image paths provided".into());
        }

        let size = image_paths.len();

        Ok(Self {
            image_paths,
            size,
            channels: 3,
            height,
            width,
            augmentation: None,
        })
    }

    pub fn with_augmentation(mut self, aug: Augmentation) -> Self {
        self.augmentation = Some(aug);
        self
    }

    pub fn get<B: Backend>(
        &self,
        idx: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
        if idx >= self.size {
            return Err(format!("Index {} out of bounds (size: {})", idx, self.size).into());
        }

        let mut img = image::open(&self.image_paths[idx])?;

        if let Some(ref aug) = self.augmentation {
            img = aug.apply(img);
        }

        img = img.resize_exact(
            self.width as u32,
            self.height as u32,
            image::imageops::FilterType::Lanczos3,
        );

        let img = img.to_rgb8();

        let mut data = vec![Vec::with_capacity(self.height * self.width); 3];
        for p in img.pixels() {
            data[0].push((p[0] as f32 / 255.0) * 2.0 - 1.0);
            data[1].push((p[1] as f32 / 255.0) * 2.0 - 1.0);
            data[2].push((p[2] as f32 / 255.0) * 2.0 - 1.0);
        }

        let flat: Vec<f32> = data.into_iter().flatten().collect();
        let tensor = Tensor::<B, 1>::from_floats(flat.as_slice(), device);
        let tensor = tensor.reshape([1, self.channels, self.height, self.width]);
        Ok(tensor)
    }

    pub fn get_batch<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
        let mut batch_images = Vec::new();

        for &idx in indices {
            batch_images.push(self.get::<B>(idx, device)?);
        }

        Ok(Tensor::cat(batch_images, 0))
    }
}

// Hyperparameters structure
#[derive(Debug, Clone, Copy)]
pub struct Hyperparameters {
    pub learning_rate: f64,
    pub kl_weight: f64,
    pub batch_size: usize,
    pub latent_dim: usize,
    pub num_timesteps: usize,
    pub vae_epochs: usize,
    pub diffusion_epochs: usize,
}

impl Hyperparameters {
    pub fn from_array(arr: &[f64]) -> Self {
        Self {
            learning_rate: arr[0],
            kl_weight: arr[1],
            batch_size: arr[2].round() as usize,
            latent_dim: arr[3].round() as usize,
            num_timesteps: arr[4].round() as usize,
            vae_epochs: arr[5].round() as usize,
            diffusion_epochs: arr[6].round() as usize,
        }
    }

    pub fn to_array(&self) -> Vec<f64> {
        vec![
            self.learning_rate,
            self.kl_weight,
            self.batch_size as f64,
            self.latent_dim as f64,
            self.num_timesteps as f64,
            self.vae_epochs as f64,
            self.diffusion_epochs as f64,
        ]
    }
}

// Bayesian Optimization Structure
pub struct BayesianOptimizer {
    pub n_iter: usize,
    pub n_initial_points: usize,
}

impl BayesianOptimizer {
    pub fn new(n_iter: usize, n_initial_points: usize) -> Self {
        Self {
            n_iter,
            n_initial_points,
        }
    }

    // Optimize all hyperparameters
    pub fn optimize<F>(&self, objective: F) -> Hyperparameters
    where
        F: Fn(&[f64]) -> f64,
    {
        println!("\n=== Starting Bayesian Optimization ===");
        println!("Optimizing 7 hyperparameters:");
        println!("  1. Learning Rate (1e-6 to 1e-3)");
        println!("  2. KL Weight (0.001 to 0.1)");
        println!("  3. Batch Size (16 to 128)");
        println!("  4. Latent Dimension (4 to 16)");
        println!("  5. Diffusion Timesteps (100 to 1000)");
        println!("  6. VAE Epochs (50 to 500)");
        println!("  7. Diffusion Epochs (50 to 300)");
        
        // Define search space for all hyperparameters
        let xlimits = array![
            [1e-6, 1e-3],      // learning_rate
            [0.001, 0.1],      // kl_weight
            [16.0, 128.0],     // batch_size
            [4.0, 16.0],       // latent_dim
            [100.0, 1000.0],   // num_timesteps
            [50.0, 500.0],     // vae_epochs
            [50.0, 300.0]      // diffusion_epochs
        ];
        
        // Create initial sampling using Latin Hypercube Sampling
        let doe = Lhs::new(&xlimits).sample(self.n_initial_points);
        
        // Evaluate initial points
        let mut x_data = doe.clone();
        let mut y_data: Array2<f64> = Array2::zeros((self.n_initial_points, 1));
        
        for i in 0..self.n_initial_points {
            let params = doe.row(i).to_vec();
            let hyperparams = Hyperparameters::from_array(&params);
            
            println!("\n--- Initial Point {}/{} ---", i + 1, self.n_initial_points);
            println!("  lr={:.6}, kl={:.4}, batch={}, latent={}, timesteps={}, vae_ep={}, diff_ep={}", 
                     hyperparams.learning_rate,
                     hyperparams.kl_weight,
                     hyperparams.batch_size,
                     hyperparams.latent_dim,
                     hyperparams.num_timesteps,
                     hyperparams.vae_epochs,
                     hyperparams.diffusion_epochs);
            
            let loss = objective(&params);
            y_data[[i, 0]] = loss;
            println!("  Loss: {:.4}", loss);
        }
        
        // Build and iterate with Gaussian Process
        for iter in 0..self.n_iter {
            println!("\n=== BO Iteration {}/{} ===", iter + 1, self.n_iter);
            
            // Fit GP model
            let gp = GpMixture::params()
                .n_clusters(1)
                .fit(&x_data, &y_data)
                .expect("GP fitting failed");
            
            // Find next point using Expected Improvement
            let egor = EgorBuilder::optimize(
                |x: &Array2<f64>| -> Array2<f64> {
                    gp.predict(x).expect("Prediction failed")
                })
                .configure(|config| {
                    config
                        .doe(&x_data)
                        .infill_strategy(InfillStrategy::EI)
                        .infill_optimizer(InfillOptimizer::Cobyla)
                        .max_iters(1)
                })
                .xlimits(&xlimits)
                .run()
                .expect("Optimization failed");
            
            let x_opt = egor.x_opt.row(0).to_vec();
            let hyperparams = Hyperparameters::from_array(&x_opt);
            
            println!("  Next evaluation:");
            println!("    lr={:.6}, kl={:.4}, batch={}, latent={}, timesteps={}, vae_ep={}, diff_ep={}", 
                     hyperparams.learning_rate,
                     hyperparams.kl_weight,
                     hyperparams.batch_size,
                     hyperparams.latent_dim,
                     hyperparams.num_timesteps,
                     hyperparams.vae_epochs,
                     hyperparams.diffusion_epochs);
            
            let y_opt = objective(&x_opt);
            println!("    Loss: {:.4}", y_opt);
            
            // Add new point to dataset
            let new_x = Array2::from_shape_vec((1, 7), x_opt).expect("Shape error");
            let new_y = array![[y_opt]];
            
            x_data = ndarray::concatenate![ndarray::Axis(0), x_data, new_x];
            y_data = ndarray::concatenate![ndarray::Axis(0), y_data, new_y];
        }
        
        // Find best parameters
        let best_idx = y_data.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        let best_params: Vec<f64> = (0..7).map(|j| x_data[[best_idx, j]]).collect();
        let best_hyperparams = Hyperparameters::from_array(&best_params);
        let best_loss = y_data[[best_idx, 0]];
        
        println!("\n=== Optimization Complete ===");
        println!("Best hyperparameters found:");
        println!("  Learning Rate: {:.6}", best_hyperparams.learning_rate);
        println!("  KL Weight: {:.4}", best_hyperparams.kl_weight);
        println!("  Batch Size: {}", best_hyperparams.batch_size);
        println!("  Latent Dimension: {}", best_hyperparams.latent_dim);
        println!("  Diffusion Timesteps: {}", best_hyperparams.num_timesteps);
        println!("  VAE Epochs: {}", best_hyperparams.vae_epochs);
        println!("  Diffusion Epochs: {}", best_hyperparams.diffusion_epochs);
        println!("  Best Loss: {:.4}", best_loss);
        
        best_hyperparams
    }
}

fn main() {
    use burn::backend::wgpu::WgpuDevice;

    println!("Initializing GPU...");
    type Backend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let in_channels = 3;

    let augmentation = Augmentation::new()
        .with_horizontal(true)
        .with_vertical(false)
        .with_rotation(true)
        .with_brightness(0.2)
        .with_contrast(0.2);

    let dataset = Image::directory(r"Pictures", 64, 64)
        .unwrap()
        .with_augmentation(augmentation);

    println!("Loaded dataset with {} images (with augmentation)", dataset.size);

    // Bayesian Optimization for ALL hyperparameters
    println!("\n🔍 Starting comprehensive Bayesian Optimization...");
    let bayesian_opt = BayesianOptimizer::new(15, 8); // 15 iterations, 8 initial points
    
    let best_hyperparams = bayesian_opt.optimize(|params| {
        let hp = Hyperparameters::from_array(params);
        
        // Validate constraints
        if hp.batch_size > dataset.size {
            return 1e6; // Penalty for invalid batch size
        }
        
        // Quick validation run with reduced epochs
        let eval_vae_epochs = (hp.vae_epochs / 10).max(2);
        let eval_batch_size = hp.batch_size.min(32); // Use smaller batches for faster eval
        
        let mut temp_model = DiffusionModel::<Backend>::new(
            &device, hp.latent_dim, in_channels, hp.num_timesteps
        );
        let optimizer_config = AdamConfig::new();
        let mut temp_optimizer = optimizer_config.init();
        
        let mut total_loss = 0.0;
        let mut count = 0;
        
        // Evaluate with reduced epochs
        for epoch in 0..eval_vae_epochs {
            let num_batches = (dataset.size + eval_batch_size - 1) / eval_batch_size;
            
            for batch_idx in 0..num_batches.min(3) { // Use only first 3 batches
                let start_idx = batch_idx * eval_batch_size;
                let end_idx = (start_idx + eval_batch_size).min(dataset.size);
                
                if end_idx - start_idx < eval_batch_size {
                    break;
                }
                
                let indices: Vec<usize> = (start_idx..end_idx).collect();
                
                if let Ok(images) = dataset.get_batch::<Backend>(&indices, &device) {
                    let kl_weight = hp.kl_weight as f32 * (1.0 + epoch as f32 / eval_vae_epochs as f32);
                    let tensor_loss = temp_model.train_vae(images, kl_weight);
                    let grads = tensor_loss.backward();
                    let grad_params = GradientsParams::from_grads(grads, &temp_model);
                    temp_model = temp_optimizer.step(hp.learning_rate, temp_model, grad_params);
                    
                    let loss_val = tensor_loss.into_scalar().elem::<f32>();
                    total_loss += loss_val;
                    count += 1;
                }
            }
        }
        
        if count == 0 {
            1e6 // Penalty if no training happened
        } else {
            (total_loss / count as f32) as f64
        }
    });

    println!("\n✅ Using optimized hyperparameters for full training:");
    println!("  Learning Rate: {:.6}", best_hyperparams.learning_rate);
    println!("  KL Weight: {:.4}", best_hyperparams.kl_weight);
    println!("  Batch Size: {}", best_hyperparams.batch_size);
    println!("  Latent Dimension: {}", best_hyperparams.latent_dim);
    println!("  Diffusion Timesteps: {}", best_hyperparams.num_timesteps);
    println!("  VAE Epochs: {}", best_hyperparams.vae_epochs);
    println!("  Diffusion Epochs: {}", best_hyperparams.diffusion_epochs);

    // Create model with optimized hyperparameters
    let mut model = DiffusionModel::<Backend>::new(
        &device,
        best_hyperparams.latent_dim,
        in_channels,
        best_hyperparams.num_timesteps
    );
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init();

    println!("\n📚 Training VAE with optimized parameters...");

    for epoch in 0..best_hyperparams.vae_epochs {
        let mut total_loss_vae = 0.0f32;
        let num_batches = (dataset.size + best_hyperparams.batch_size - 1) / best_hyperparams.batch_size;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * best_hyperparams.batch_size;
            let end_idx = (start_idx + best_hyperparams.batch_size).min(dataset.size);

            if end_idx - start_idx < best_hyperparams.batch_size {
                break;
            }

            let indices: Vec<usize> = (start_idx..end_idx).collect();

            if let Ok(images) = dataset.get_batch::<Backend>(&indices, &device) {
                let kl_weight = best_hyperparams.kl_weight as f32 
                    * (1.0 + epoch as f32 / best_hyperparams.vae_epochs as f32);

                let tensor_loss = model.train_vae(images.clone(), kl_weight);

                let grads = tensor_loss.backward();
                let grad_params = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(best_hyperparams.learning_rate, model, grad_params);

                let loss_val = tensor_loss.into_scalar().elem::<f32>();
                total_loss_vae += loss_val;

                // Save VAE output periodically
                if epoch % 10 == 0 && batch_idx == 0 {
                    let reconstructed = model.vae.decode_from_mean(images.clone());
                    if let Ok(_) = save_tensor_as_image(
                        reconstructed,
                        &format!("vae_output_epoch{}.png", epoch)
                    ) {
                        println!("  💾 Saved VAE output for epoch {}", epoch);
                    }
                }

                if batch_idx % 10 == 0 {
                    println!(
                        "  Batch {}/{}: Loss = {:.4}",
                        batch_idx, num_batches, loss_val
                    );
                }
            }
        }

        let average_loss_vae = total_loss_vae / num_batches as f32;
        println!(
            "Epoch {}/{}: Average VAE Loss = {:.4}",
            epoch + 1,
            best_hyperparams.vae_epochs,
            average_loss_vae
        );
    }

    println!("\n✅ VAE training complete!");
    save_model(&model, "VAE_optimized.bin").unwrap();

    println!("\n🎨 Training Diffusion Model...");

    for epoch in 0..best_hyperparams.diffusion_epochs {
        let mut total_loss = 0.0f32;
        let num_batches = (dataset.size + best_hyperparams.batch_size - 1) / best_hyperparams.batch_size;

        println!("\nEpoch {}/{}", epoch + 1, best_hyperparams.diffusion_epochs);

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * best_hyperparams.batch_size;
            let end_idx = (start_idx + best_hyperparams.batch_size).min(dataset.size);
            
            if end_idx - start_idx < best_hyperparams.batch_size {
                break;
            }
            
            let indices: Vec<usize> = (start_idx..end_idx).collect();

            if let Ok(images) = dataset.get_batch::<Backend>(&indices, &device) {
                let tensor_loss = model.forward(images, &device);

                let grads = tensor_loss.backward();
                let grad_params = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(best_hyperparams.learning_rate, model, grad_params);

                let loss_val = tensor_loss.into_scalar().elem::<f32>();
                total_loss += loss_val;

                if batch_idx % 10 == 0 {
                    println!(
                        "  Batch {}/{}: Loss = {:.4}",
                        batch_idx, num_batches, loss_val
                    );
                }
            }
        }

        let average_loss = total_loss / num_batches as f32;
        println!(
            "Epoch {}/{}: Average Diffusion Loss = {:.4}",
            epoch + 1,
            best_hyperparams.diffusion_epochs,
            average_loss
        );

        // Generate samples periodically
        if (epoch + 1) % 10 == 0 {
            println!("\n🖼️  Generating sample images...");
            let generated = model.sample(4, &device);
            
            for i in 0..4 {
                let single_image = generated.clone().slice([i..i + 1]);
                if let Ok(_) = save_tensor_as_image(
                    single_image,
                    &format!("generated_epoch{}_sample{}.png", epoch + 1, i)
                ) {
                    println!("  💾 Saved sample {} for epoch {}", i, epoch + 1);
                }
            }
        }
    }

    println!("\n✅ Diffusion model training complete!");
    save_model(&model, "DiffusionModel_optimized.bin").unwrap();

    // Final generation
    println!("\n🎉 Generating final samples...");
    let final_samples = model.sample(8, &device);
    
    for i in 0..8 {
        let single_image = final_samples.clone().slice([i..i + 1]);
        if let Ok(_) = save_tensor_as_image(
            single_image,
            &format!("final_sample_{}.png", i)
        ) {
            println!("  💾 Saved final sample {}", i);
        }
    }

    println!("\n🎊 Training and generation complete!");
    println!("📁 Check your directory for:");
    println!("   - VAE_optimized.bin");
    println!("   - DiffusionModel_optimized.bin");
    println!("   - vae_output_epoch*.png");
    println!("   - generated_epoch*_sample*.png");
    println!("   - final_sample_*.png");
}