use burn::backend::{Autodiff, Wgpu};
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig, PaddingConfig2d};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::ElementConversion;
use burn::tensor::Int;
use burn::tensor::Tensor;
use image::{DynamicImage, ImageBuffer, Rgb};
use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyList;

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
        let variance = variance.clamp(-20.0, 2.0);
        let std = (variance.clone() * 0.5).exp();
        let std = std + 1e-8;
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

//adding attention mechanism to improve the efficiency
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

            // Downsampling path with stride=2
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
        let beta_start = 1e-4_f32;
        let beta_end = 0.02_f32;
        let steps = self.num_time as usize;

        let betas: Vec<f32> = (0..steps)
            .map(|i| {
                let ratio = i as f32 / (steps - 1).max(1) as f32;
                let beta = beta_start + (beta_end - beta_start) * ratio;
                beta.clamp(1e-6, 0.999) // Prevent extreme values
            })
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

        // Gather alpha values for the batch (minimize CPU operations)
        let time_values: Vec<i32> = time.to_data().to_vec().unwrap();
        let alpha_values: Vec<f32> = alpha.to_data().to_vec().unwrap();

        let gathered_alphas: Vec<f32> = time_values
            .iter()
            .map(|&t| {
                let idx = (t as usize).min(alpha_values.len() - 1);
                let alpha_val = alpha_values[idx];
                if alpha_val.is_nan() || alpha_val.is_infinite() || alpha_val <= 0.0 {
                    1e-6_f32 // Safe fallback
                } else {
                    alpha_val.clamp(1e-8, 0.9999)
                }
            })
            .collect();

        let alpha_t: Tensor<B, 1> = Tensor::from_floats(gathered_alphas.as_slice(), device);
        let alpha_t = alpha_t
            .reshape([batch_size, 1, 1, 1])
            .repeat_dim(1, channels)
            .repeat_dim(2, height)
            .repeat_dim(3, width);

        let sqrt_alpha = alpha_t.clone().sqrt().clamp(1e-8, 1.0);
        let one_minus_alpha = Tensor::ones_like(&alpha_t) - alpha_t;
        let sqrt_one_minus_alpha = one_minus_alpha.clamp(1e-8, 1.0).sqrt();

        // All operations stay on GPU with safety checks
        let noisy = x * sqrt_alpha + noise.clone() * sqrt_one_minus_alpha;

        (noisy, noise)
    }
    pub fn noise_predict(&self, x_t: Tensor<B, 4>, time_emb: Tensor<B, 2>) -> Tensor<B, 4> {
        self.unet.forward(x_t, time_emb)
    }

    pub fn forward(&self, x: Tensor<B, 4>, device: &B::Device) -> Tensor<B, 1> {
        let z = self.vae.encode(x.clone()).detach();
        let batch_size = x.dims()[0];

        let t_values: Vec<i32> = (0..batch_size)
            .map(|_| (rand::random::<u32>() % self.num_time as u32) as i32)
            .collect();

        let t = Tensor::from_ints(t_values.as_slice(), device);
        let (z_noisy, noise) = self.add_noise(z, t.clone(), device);

        // Check for NaN in noisy latent
        let z_sample = z_noisy.clone().into_scalar().elem::<f32>();
        if z_sample.is_nan() || z_sample.is_infinite() {
            let tensor: Tensor<B, 1> = Tensor::from_floats([100.0], device);
            return tensor.reshape([1]);
        }

        let t_emb = sin_time_addition(device, t, 256);

        let noise_pred = self.noise_predict(z_noisy, t_emb);

        // Check for NaN in noise prediction
        let pred_sample = noise_pred.clone().into_scalar().elem::<f32>();
        if pred_sample.is_nan() || pred_sample.is_infinite() {
            let tensor: Tensor<B, 1> = Tensor::from_floats([100.0], device);
            return tensor.reshape([1]);
        }

        let noise_pred_2d: Tensor<B, 2> = noise_pred.flatten(1, 3);
        let noise_2d: Tensor<B, 2> = noise.flatten(1, 3);

        let diff = noise_pred_2d - noise_2d;
        let squared = diff.powf_scalar(2.0);
        let diffusion_loss = squared.mean();

        // Final safety check
        let loss_val = diffusion_loss.clone().into_scalar().elem::<f32>();
        if loss_val.is_nan() || loss_val.is_infinite() {
            let tensor: Tensor<B, 1> = Tensor::from_floats([100.0], device);
            return tensor.reshape([1]);
        }

        diffusion_loss.reshape([1])
    }

    pub fn denoising_process(&self, batch_size: usize, device: &B::Device) -> Tensor<B, 4> {
        let mut z: Tensor<B, 4> = Tensor::random(
            [batch_size, self.latent_dimen, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );

        // Simple NaN check for initial tensor
        let z_check = z.clone().into_scalar().elem::<f32>();
        if z_check.is_nan() || z_check.is_infinite() {
            println!("NaN detected in initial noise tensor. Using zeros.");
            z = Tensor::zeros([batch_size, self.latent_dimen, 4, 4], device);
        }

        //Denoising The Added NOises using DDPM denoising formula
        let beta = self.get_betas(device);
        let alpha_1 = self.get_alphas(device);

        for i in (0..self.num_time).rev() {
            let i_tensor = Tensor::from_ints(vec![i as i32; batch_size].as_slice(), device);
            let i_emb = sin_time_addition(device, i_tensor, 256);

            let noise_pred = self.noise_predict(z.clone(), i_emb);

            // Simple NaN check for noise prediction
            let noise_check = noise_pred.clone().into_scalar().elem::<f32>();
            if noise_check.is_nan() || noise_check.is_infinite() {
                println!("NaN detected in noise prediction. Breaking iteration.");
                break;
            }

            let alpha_t = alpha_1.clone().slice([i..i + 1]);
            let beta_t = beta.clone().slice([i..i + 1]);

            let alpha_sqrt = alpha_t.clone().sqrt();
            let alpha_sqrt_1 = (alpha_t.clone() * (-1.0) + 1.0).sqrt();

            let [b, c, h, w] = z.dims();
            let alpha_sqrt_exp = alpha_sqrt.reshape([1, 1, 1, 1]).expand([b, c, h, w]);
            let alpha_sqrt_1_exp = alpha_sqrt_1.reshape([1, 1, 1, 1]).expand([b, c, h, w]);

            let pred_x0 = (z.clone() - noise_pred.clone() * alpha_sqrt_1_exp) / alpha_sqrt_exp;

            // Simple NaN check for predicted x0
            let pred_x0_check = pred_x0.clone().into_scalar().elem::<f32>();
            if pred_x0_check.is_nan() || pred_x0_check.is_infinite() {
                println!("NaN detected in predicted x0. Breaking iteration.");
                break;
            }

            if i > 0 {
                let alpha_prev = alpha_1.clone().slice([i - 1..i]);

                let one_minus_alpha_t = alpha_t.clone() * (-1.0) + 1.0;
                let one_minus_alpha_prev = alpha_prev.clone() * (-1.0) + 1.0;

                let variance = (one_minus_alpha_prev.clone() / (one_minus_alpha_t.clone() + 1e-8))
                    * beta_t.clone();
                let sigma_t = variance.sqrt();
                let sigma_t_exp = sigma_t.reshape([1, 1, 1, 1]).expand([b, c, h, w]);

                let coef1 =
                    alpha_prev.clone().sqrt() * beta_t.clone() / (one_minus_alpha_t.clone() + 1e-8);

                let one_minus_beta: Tensor<B, 1> = 1.0 - beta_t.clone();
                let coef2 =
                    one_minus_beta.sqrt() * one_minus_alpha_prev / (one_minus_alpha_t + 1e-8);

                let coef1_expanded = coef1.reshape([1, 1, 1, 1]).expand([b, c, h, w]);
                let coef2_expanded = coef2.reshape([1, 1, 1, 1]).expand([b, c, h, w]);

                let mean = pred_x0.clone() * coef1_expanded + z.clone() * coef2_expanded;

                let noise = Tensor::random_like(&z, burn::tensor::Distribution::Normal(0.0, 1.0));

                z = mean + sigma_t_exp * noise;

                // Simple NaN check for z after update
                let z_check = z.clone().into_scalar().elem::<f32>();
                if z_check.is_nan() || z_check.is_infinite() {
                    println!("NaN detected in z update. Breaking iteration.");
                    break;
                }
            } else {
                z = pred_x0;

                let z_final_check = z.clone().into_scalar().elem::<f32>();
                if z_final_check.is_nan() || z_final_check.is_infinite() {
                    println!("NaN detected in final z. Breaking iteration.");
                    break;
                }
            }

            if i % 25 == 0 {
                println!("  Denoising step {}/{}", self.num_time - i, self.num_time);
            }
        }

        // Final NaN check before decoding
        let final_z_check = z.clone().into_scalar().elem::<f32>();
        if final_z_check.is_nan() || final_z_check.is_infinite() {
            println!("NaN detected before decoding. Using fallback tensor.");
            z = Tensor::zeros([batch_size, self.latent_dimen, 4, 4], device);
        }

        let decoded = self.vae.decode(z);

        let decoded_check = decoded.clone().into_scalar().elem::<f32>();
        if decoded_check.is_nan() || decoded_check.is_infinite() {
            println!("NaN detected in decoded output. Returning zeros.");
            return Tensor::zeros([batch_size, 3, 64, 64], device);
        }

        decoded
    }

    pub fn train_vae(&self, x: Tensor<B, 4>, kl_weight: f32) -> Tensor<B, 1> {
        let (reconstruction, mean, variance) = self.vae.forward(x.clone());

        let epsilon = 1e-8;
        let variance_safe = variance.clone() + epsilon;

        // More aggressive clamping to prevent exp() explosion
        let variance_crt = variance_safe.clamp(-10.0, 1.0);

        // Check for NaN/inf in mean and variance before computation
        let mean_data = mean.clone().into_scalar().elem::<f32>();
        let var_sample = variance_crt.clone().into_scalar().elem::<f32>();

        if mean_data.is_nan()
            || mean_data.is_infinite()
            || var_sample.is_nan()
            || var_sample.is_infinite()
        {
            return Tensor::from_floats([100.0], &x.device());
        }

        let kl_loss: Tensor<B, 1> = -0.5
            * (Tensor::ones_like(&variance_crt) + variance_crt.clone()
                - mean.clone().clamp(-5.0, 5.0).powf_scalar(2.0)
                - variance_crt.exp().clamp(0.0, 100.0))
            .mean();

        let recon_flat: Tensor<B, 2> = reconstruction.flatten(1, 3);
        let input_flat: Tensor<B, 2> = x.flatten(1, 3);
        let recon_loss = burn::nn::loss::MseLoss::new().forward(
            recon_flat,
            input_flat,
            burn::nn::loss::Reduction::Mean,
        );

        let kl_loss_1d: Tensor<B, 1> = kl_loss.mul_scalar(kl_weight.clamp(0.0, 1.0)).reshape([1]);
        let total_loss = recon_loss + kl_loss_1d;

        // Final NaN check
        let loss_val = total_loss.clone().into_scalar().elem::<f32>();
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
            vertical_flip: true,
            rotation: true,
            brightness: Some(0.2),
            contrast: Some(0.2),
        }
    }

    pub fn horizontal(mut self, flipped: bool) -> Self {
        self.horizontal_flip = flipped;
        self
    }

    pub fn vertical(mut self, flipped: bool) -> Self {
        self.vertical_flip = flipped;
        self
    }

    pub fn rotation(mut self, flipped: bool) -> Self {
        self.rotation = flipped;
        self
    }

    pub fn brightness(mut self, range: f32) -> Self {
        self.brightness = Some(range);
        self
    }

    pub fn contrast(mut self, range: f32) -> Self {
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
            let rotation = rand::random::<u32>() % 4;
            for _ in 0..rotation {
                img = img.rotate90();
            }
        }

        if let Some(brightness_range) = self.brightness {
            let factor = 1.0 + (rand::random::<f32>() * 2.0 - 1.0) * brightness_range;
            img = adjust_brightness(img, factor);
        }
        if let Some(contrast_range) = self.contrast {
            let factor = 1.0 + (rand::random::<f32>() * 2.0 - 1.0) * contrast_range;
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

    // Calculate mean pixel value
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

        // Load image as DynamicImage
        let mut img = image::open(&self.image_paths[idx])?;

        // Apply augmentation (if any)
        if let Some(ref aug) = self.augmentation {
            img = aug.apply(img);
        }

        // Resize (still DynamicImage)
        img = img.resize_exact(
            self.width as u32,
            self.height as u32,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to RGB8 for tensorization
        let img = img.to_rgb8();

        // Convert to CHW tensor
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

#[derive(Debug, Clone)]
pub struct Hyperparameters {
    pub learning_rate: f64,
    pub kl_loss_weight: f64,
    pub batch_size: usize,
    pub latent_dimen: usize,
    pub num_steps: usize,
    pub vae_epochs: usize,
    pub num_epochs: usize,
}

impl Hyperparameters {
    pub fn array(params: &[f64]) -> Self {
        Self {
            learning_rate: params[0],
            kl_loss_weight: params[1],
            batch_size: params[2] as usize,
            latent_dimen: params[3] as usize,
            num_steps: params[4] as usize,
            vae_epochs: params[5] as usize,
            num_epochs: params[6] as usize,
        }
    }
}

pub struct Bayesian {
    py_optimizer: Py<PyAny>,
    n_iterations: usize,
}

impl Bayesian {
    fn new(bounds: Vec<(f64, f64)>, n_iterations: usize) -> PyResult<Self> {
        Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            path.call_method1("append", ("bayesian",))?;

            let bo_module = py.import("bayesian")?;
            let bo_class = bo_module.getattr("BayesianLDM")?;
            let py_bounds = PyList::empty(py);
            for bound in &bounds {
                py_bounds.push(bound.into_py(py))?;
            }

            let py_optimizer = bo_class.call((py_bounds, n_iterations), None)?;

            Ok(Self {
                py_optimizer: py_optimizer.into(),
                n_iterations,
            })
        })
    }

    fn suggest(&self) -> PyResult<Vec<f64>> {
        Python::with_gil(|py| {
            let optimizer = self.py_optimizer.as_ref(py);
            let result = optimizer.call_method0("suggest")?;
            result.extract()
        })
    }

    fn observe(&self, params: Vec<f64>, score: f64) -> PyResult<()> {
        Python::with_gil(|py| {
            let optimizer = self.py_optimizer.as_ref(py);
            let py_params = PyList::empty(py);
            for param in &params {
                py_params.append((*param).into_py(py))?;
            }
            optimizer.call_method("observe", (py_params, score), None)?;
            Ok(())
        })
    }

    fn get_best(&self) -> PyResult<(Vec<f64>, f64)> {
        Python::with_gil(|py| {
            let optimizer = self.py_optimizer.as_ref(py);
            let result = optimizer.call_method0("get_best")?;

            let best_params: Vec<f64> = result.get_item(0)?.extract()?;
            let best_score: f64 = result.get_item(1)?.extract()?;

            Ok((best_params, best_score))
        })
    }

    fn optimize<F>(&self, mut f: F) -> PyResult<Hyperparameters>
    where
        F: FnMut(&[f64]) -> f64,
    {
        for _ in 0..self.n_iterations {
            let params = self.suggest()?;
            let score = f(&params);
            self.observe(params, score)?;
        }
        let (best_params, _) = self.get_best()?;
        Ok(Hyperparameters::array(&best_params))
    }
}

pub fn train_ldm_epoch<B: Backend>(
    model: &DiffusionModel<B>,
    dataset: &Image,
    device: &B::Device,
    batch_size: usize,
    _learning_rate: f64,
) -> f32 {
    let mut total_loss = 0.0f32;
    let mut valid_batches = 0;
    let num_batches = dataset.size / batch_size;

    for batch_idx in 0..num_batches {
        let mut batch_images = Vec::new();

        for i in 0..batch_size {
            let idx = batch_idx * batch_size + i;
            if let Ok(img) = dataset.get::<B>(idx, device) {
                batch_images.push(img);
            }
        }

        if !batch_images.is_empty() {
            let images = Tensor::cat(batch_images, 0);
            let loss = model.forward(images, device);
            let loss_val = loss.into_scalar().elem::<f32>();

            // Simple NaN check - break immediately
            if loss_val.is_nan() || loss_val.is_infinite() || loss_val > 150.0 {
                println!("NaN detected in train_ldm_epoch. Stopping.");
                break;
            }

            total_loss += loss_val;
            valid_batches += 1;
        }
    }

    if valid_batches == 0 {
        println!("Warning: No valid batches processed in train_ldm_epoch");
        return 100.0; // Return high loss value
    }

    let avg_loss = total_loss / valid_batches as f32;

    // Final NaN check for average
    if avg_loss.is_nan() || avg_loss.is_infinite() {
        println!("NaN detected in train_ldm_epoch average. Returning high loss.");
        return 100.0;
    }

    avg_loss
}

pub fn save_as_image<B: Backend>(
    tensor: Tensor<B, 4>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let [_batch, _channels, height, width] = tensor.dims();

    // Check tensor for NaN/Inf before processing
    let tensor_check = tensor.clone().into_scalar().elem::<f32>();
    let values: Vec<f32> = if tensor_check.is_nan() || tensor_check.is_infinite() {
        println!(
            "Warning: NaN/Inf detected in tensor for image {}. Creating black image.",
            path
        );
        // Return error for NaN tensors
        return Err("NaN detected in tensor".into());
    } else {
        let img_tensor = tensor.slice([0..1]);
        img_tensor
            .to_data()
            .to_vec()
            .expect("Failed to convert tensor data")
    };

    let img = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
        let pixel_idx = (y as usize * width + x as usize) as usize;

        let r_idx = pixel_idx;
        let g_idx = pixel_idx + (height * width);
        let b_idx = pixel_idx + 2 * (height * width);

        let r = ((values[r_idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
        let g = ((values[g_idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;
        let b = ((values[b_idx] + 1.0) * 127.5).clamp(0.0, 255.0) as u8;

        Rgb([r, g, b])
    });

    img.save(path)?;
    Ok(())
}
fn main() {
    use burn::backend::wgpu::WgpuDevice;

    println!("Initializing GPU...");
    type Backend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let augmentation = Augmentation::new()
        .horizontal(true)
        .vertical(false)
        .rotation(true)
        .brightness(0.2)
        .contrast(0.2);

    // Load dataset with augmentation
    let dataset = Image::directory(r"Pictures", 64, 64)
        .unwrap()
        .with_augmentation(augmentation);

    println!(
        "Starting training with {} images (with augmentation)...",
        dataset.size
    );

    let bounds = vec![
        (0.0001, 0.1),
        (0.001, 0.1),
        (16.0, 64.0),
        (12.0, 64.0),
        (150.0, 1200.0),
        (30.0, 150.0),
        (50.0, 200.0),
    ];
    let bayesian_opt = Bayesian::new(bounds, 10).unwrap();

    let best_hyperparameter = bayesian_opt
        .optimize(|params: &[f64]| {
            let hp = Hyperparameters::array(params);

            let opt_run_id: u64 = rand::random();
            let opt_run_id_str = format!("{:x}", opt_run_id);

            if hp.batch_size > dataset.size || hp.batch_size < 4 {
                println!(
                    " Invalid batch size: {} (dataset: {})",
                    hp.batch_size, dataset.size
                );
                return 100.0;
            }

            // Simple NaN detection flag
            let mut nan_detected = false;

            let eval_vae_epochs = hp.vae_epochs;
            let eval_batch_size = hp.batch_size.min(32);

            let eval_diffusion_epochs = hp.num_epochs;
            let eval_diffusion_batches = hp.batch_size.min(32);

            let mut temp_model =
                DiffusionModel::<Backend>::new(&device, hp.latent_dimen, 3, hp.num_steps);

            let optimizer = AdamConfig::new();
            let mut temp_optimizer = optimizer.init();

            let mut vae_total_loss = 0.0;
            let mut vae_count = 0;

            let mut last_images: Option<Tensor<Backend, 4>> = None;

            for epoch in 0..eval_vae_epochs {
                println!("\n Epochs {}/{}", epoch + 1, eval_vae_epochs);
                let num_batches = (dataset.size + eval_batch_size - 1) / eval_batch_size;

                for batch_idx in 0..num_batches {
                    let start_idx = batch_idx * eval_batch_size;
                    let end_idx = (start_idx + eval_batch_size).min(dataset.size);

                    if end_idx - start_idx < eval_batch_size {
                        break;
                    }

                    let indices: Vec<usize> = (start_idx..end_idx).collect();

                    if let Ok(images) = dataset.get_batch::<Backend>(&indices, &device) {
                        if batch_idx == 0 {
                            last_images = Some(images.clone());
                        }
                        let kl_weight =
                            hp.kl_loss_weight * (1.0 + epoch as f64 / eval_vae_epochs as f64);

                        let tensor_loss = temp_model.train_vae(images, kl_weight as f32);

                        let loss_val = tensor_loss.clone().into_scalar().elem::<f32>();

                        // Simple NaN check - break immediately
                        if loss_val.is_nan() || loss_val.is_infinite() {
                            println!(" NaN detected in VAE batch. Breaking iteration.");
                            break;
                        }

                        // Gradient clipping before step
                        let grads = tensor_loss.backward();
                        let grad_params = GradientsParams::from_grads(grads, &temp_model);

                        // Use smaller learning rate if loss is getting large
                        let adaptive_lr = if loss_val > 10.0 {
                            hp.learning_rate * 0.1
                        } else {
                            hp.learning_rate
                        };

                        temp_model = temp_optimizer.step(adaptive_lr, temp_model, grad_params);

                        vae_total_loss += loss_val;
                        vae_count += 1;

                        if batch_idx % 10 == 0 {
                            println!(
                                "  Batch {}/{}: Loss = {:.4}",
                                batch_idx, num_batches, loss_val
                            );
                        }
                    }
                }

                // Break out of epoch loop if NaN detected
                if nan_detected {
                    break;
                }

                let avg_epoch_loss = vae_total_loss / vae_count as f32;
                println!(
                    "  Eval Epoch {}/{}: Avg Loss = {:.4}",
                    epoch + 1,
                    eval_vae_epochs,
                    avg_epoch_loss
                );

                if avg_epoch_loss.is_nan() || avg_epoch_loss.is_infinite() {
                    println!("NaN detected in epoch average. Breaking VAE training.");
                    break;
                }

                let current_epoch = epoch + 1;

                if current_epoch % 2 == 0 || current_epoch == 1 {
                    if let Some(ref images) = last_images {
                        if vae_count > 0 {
                            let z = temp_model.vae.encode(images.clone());
                            let reconstructed = temp_model.vae.decode(z);

                            let filename = format!(
                                "run{}_e{}_lr{:.0e}_kl{:.3}_bs{}_ld{}_loss{:.4}.png",
                                opt_run_id_str,
                                current_epoch,
                                hp.learning_rate,
                                hp.kl_loss_weight,
                                hp.batch_size,
                                hp.latent_dimen,
                                avg_epoch_loss
                            );
                            if let Ok(_) = save_as_image(reconstructed, &filename) {
                                println!(" Saved reconstruction: {}", filename);
                            } else {
                                eprintln!(" Failed to save image: {}", filename);
                            }
                        }
                    }
                }
            }

            if vae_count == 0 || nan_detected {
                println!(
                    " VAE training failed: vae_count={}, nan_detected={}",
                    vae_count, nan_detected
                );
                return 100.0; 
            }

            let vae_loss_ = (vae_total_loss / vae_count as f32) as f64;

            if vae_loss_.is_nan() || vae_loss_.is_infinite() || vae_loss_ > 150.0 {
                println!(
                    " Invalid final VAE loss: {}. Telling BO to skip this configuration.",
                    vae_loss_
                );
                return 100.0; // Return high loss to BO
            }
            let vae_loss = vae_loss_.clamp(0.001, 100.0);

            println!("✓ VAE Final loss: {:.6}", vae_loss);

            //Calculating Diffusion Model using Hyperparameters
            let mut diffusion_total_loss = 0.0;
            let mut diffusion_count = 0;

            println!("\n Diffusion Model Training has started with Hyperparameters....");

            for epoch in 0..eval_diffusion_epochs {
                println!("\n Epochs {}/{}", epoch + 1, eval_diffusion_epochs);
                nan_detected = false;
                let num_batches =
                    (dataset.size + eval_diffusion_batches - 1) / eval_diffusion_batches;

                for batch_idx in 0..num_batches {
                    let start_idx = batch_idx * eval_diffusion_batches;
                    let end_idx = (start_idx + eval_diffusion_batches).min(dataset.size);

                    if end_idx - start_idx < eval_batch_size {
                        break;
                    }

                    let indices: Vec<usize> = (start_idx..end_idx).collect();

                    if let Ok(images) = dataset.get_batch::<Backend>(&indices, &device) {
                        let tensor_loss = temp_model.forward(images, &device);

                        let loss_val = tensor_loss.clone().into_scalar().elem::<f32>();

                        if loss_val.is_nan() || loss_val.is_infinite() || loss_val > 150.0 {
                            println!(" NaN detected in diffusion batch. Breaking iteration.");
                            break;
                        }

                        let grads = tensor_loss.backward();
                        let grad_params = GradientsParams::from_grads(grads, &temp_model);
                        let lr = hp.learning_rate * 0.99_f64.powi(epoch as i32);
                        temp_model = temp_optimizer.step(lr, temp_model, grad_params);

                        diffusion_total_loss += loss_val;
                        diffusion_count += 1;

                        if batch_idx % 10 == 0 {
                            println!(
                                "  Batch {}/{}: Loss = {:.4}",
                                batch_idx, num_batches, loss_val
                            );
                        }
                    }
                }

                // Break out if NaN detected
                if nan_detected {
                    break;
                }

                let avg_epoch_loss_diffusion = diffusion_total_loss / diffusion_count as f32;
                println!(
                    "  Eval Epoch {}/{}: Avg Loss = {:.4}",
                    epoch + 1,
                    eval_diffusion_epochs,
                    avg_epoch_loss_diffusion
                );

                // Check epoch average for NaN
                if avg_epoch_loss_diffusion.is_nan() || avg_epoch_loss_diffusion.is_infinite() {
                    println!("NaN in diffusion epoch average. Breaking.");
                    break;
                }

                if (epoch + 1) % 10 == 0 {
                    println!("\n Generating sample images...");
                    let generated = temp_model.denoising_process(1, &device);
                    if let Ok(_) = save_as_image(
                        generated,
                        &format!("generated_epoch_diffusion_{}.png", epoch + 1),
                    ) {
                        println!(" Saved sample for epoch_diffusion {}", epoch + 1);
                    }
                }
            }

            let diffusion_loss = (diffusion_total_loss / diffusion_count as f32) as f64;

            //Final NaN check for diffusion loss
            if diffusion_loss.is_nan() || diffusion_loss.is_infinite() || diffusion_loss > 100.0 {
                println!(
                    " Invalid final diffusion loss: {}. Configuration rejected.",
                    diffusion_loss
                );
                return 100.0;
            }

            let combined_loss = vae_loss * 0.3 + diffusion_loss * 0.7;

            println!(
                "✓ VAE Loss: {:.6}, Diffusion Loss: {:.6}, Combined: {:.6}",
                vae_loss, diffusion_loss, combined_loss
            );

            -combined_loss.clamp(0.001, 100.0)
        })
        .unwrap();
    // Create model with optimized hyperparameters

    println!("\n Selected the best Hyperparameters from the choices...");
    let in_channels = 3;
    let mut model = DiffusionModel::<Backend>::new(
        &device,
        best_hyperparameter.latent_dimen,
        in_channels,
        best_hyperparameter.num_steps,
    );
    let optimizer_config = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-4)));
    let mut optimizer = optimizer_config.init();

    println!("\n Training VAE with optimized parameters...");

    for epoch in 0..best_hyperparameter.vae_epochs {
        let mut total_loss_vae = 0.0f32;
        let num_batches =
            (dataset.size + best_hyperparameter.batch_size - 1) / best_hyperparameter.batch_size;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * best_hyperparameter.batch_size;
            let end_idx = (start_idx + best_hyperparameter.batch_size).min(dataset.size);

            if end_idx - start_idx < best_hyperparameter.batch_size {
                break;
            }

            let indices: Vec<usize> = (start_idx..end_idx).collect();

            if let Ok(images) = dataset.get_batch::<Backend>(&indices, &device) {
                let kl_weight = best_hyperparameter.kl_loss_weight as f32
                    * (1.0 + epoch as f32 / best_hyperparameter.vae_epochs as f32);

                let tensor_loss = model.train_vae(images.clone(), kl_weight);

                let loss_val = tensor_loss.clone().into_scalar().elem::<f32>();

                // Simple NaN check
                if loss_val.is_nan() || loss_val.is_infinite() || loss_val > 150.0 {
                    println!("NaN detected in VAE training. Breaking.");
                    break;
                }

                let grads = tensor_loss.backward();
                let grad_params = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(best_hyperparameter.learning_rate, model, grad_params);

                total_loss_vae += loss_val;

                if epoch % 5 == 0 && batch_idx == 0 {
                    let z = model.vae.encode(images.clone());
                    let reconstructed = model.vae.decode(z);
                    if let Ok(_) =
                        save_as_image(reconstructed, &format!("vae_output_epoch{}.png", epoch))
                    {
                        println!(" Saved VAE output for epoch {}", epoch);
                    }
                }

                if batch_idx % 5 == 0 {
                    println!(
                        "  Batch {}/{}: Loss = {:.4}",
                        batch_idx, num_batches, loss_val
                    );
                }
            }
        }

        let average_loss_vae = total_loss_vae / num_batches as f32;

        // Simple NaN check for epoch average
        if average_loss_vae.is_nan() || average_loss_vae.is_infinite() || average_loss_vae > 150.0 {
            println!("NaN detected in VAE epoch average. Breaking.");
            break;
        }

        println!(
            "Epoch {}/{}: Average VAE Loss = {:.4}",
            epoch + 1,
            best_hyperparameter.vae_epochs,
            average_loss_vae
        );
    }

    println!("\n VAE training complete.....");
    save_model(&model, "VAE_optimized.bin").unwrap();

    println!("\n  Training Diffusion Model with Suitable Hyperparameters......");

    for epoch in 0..best_hyperparameter.num_epochs {
        let mut total_loss = 0.0f32;
        let num_batches =
            (dataset.size + best_hyperparameter.batch_size - 1) / best_hyperparameter.batch_size;

        println!("\n Epoch {}/{}", epoch + 1, best_hyperparameter.num_epochs);

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * best_hyperparameter.batch_size;
            let end_idx = (start_idx + best_hyperparameter.batch_size).min(dataset.size);

            if end_idx - start_idx < best_hyperparameter.batch_size {
                break;
            }

            let indices: Vec<usize> = (start_idx..end_idx).collect();

            if let Ok(images) = dataset.get_batch::<Backend>(&indices, &device) {
                let tensor_loss = model.forward(images, &device);

                let loss_val = tensor_loss.clone().into_scalar().elem::<f32>();

                // Simple NaN check
                if loss_val.is_nan() || loss_val.is_infinite() || loss_val > 100.0 {
                    println!("NaN detected in diffusion training. Breaking.");
                    break;
                }

                let grads = tensor_loss.backward();
                let grad_params = GradientsParams::from_grads(grads, &model);
                let lr = best_hyperparameter.learning_rate * 0.99_f64.powi(epoch as i32);
                model = optimizer.step(lr, model, grad_params);

                total_loss += loss_val;

                if batch_idx % 5 == 0 {
                    println!(
                        "  Batch {}/{}: Loss = {:.4}",
                        batch_idx, num_batches, loss_val
                    );
                }
            }
        }

        let average_loss = total_loss / num_batches as f32;

        // Simple NaN check for epoch average
        if average_loss.is_nan() || average_loss.is_infinite() || average_loss > 150.0 {
            println!("NaN detected in diffusion epoch average. Breaking.");
            break;
        }

        println!(
            "Epoch {}/{}: Average Diffusion Loss = {:.4}",
            epoch + 1,
            best_hyperparameter.num_epochs,
            average_loss
        );
        if (epoch + 1) % 5 == 0 {
            println!("\n Generating sample images...");
            let generated = model.denoising_process(4, &device);

            let single_image = generated.clone().slice([0..1]);
            if let Ok(_) = save_as_image(single_image, &format!("generated_epoch{}.png", epoch + 1))
            {
                println!(" Saved sample for epoch {}", epoch + 1);
            }
        }
    }

    println!("\n Diffusion model training complete!");
    save_model(&model, "DiffusionModel_optimized.bin").unwrap();

    println!("\n Generating final sample.");
    let final_sample = model.denoising_process(1, &device);

    if let Ok(_) = save_as_image(final_sample, "final_sample.png") {
        println!(" Saved final sample: final_sample.png");
    } else {
        println!(" Failed to save final sample");
    }

    println!("\n Training and generation complete....");
    println!(" \n vae output epoch.png");
    println!("\n generated epoch sample.png");
    println!("\n final_sample.png");
}
