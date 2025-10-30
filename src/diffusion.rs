use burn::backend::{Autodiff, Wgpu};
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::ElementConversion;
use burn::tensor::Int;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct TimeAddition<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> TimeAddition<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(dim, dim * 8).init(device),
            linear2: LinearConfig::new(dim * 8, dim).init(device),
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
                .init(device),
            bn1: BatchNormConfig::new(channels).init(device),

            conv2: Conv2dConfig::new([channels, channels], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
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
    down4: Conv2d<B>,
    down_res4: ResidualBlock<B>,
    
    mid_res1: ResidualBlock<B>,
    mid_res2: ResidualBlock<B>,
    
    up1: ConvTranspose2d<B>,
    up_res1: ResidualBlock<B>,
    up2: ConvTranspose2d<B>,
    up_res2: ResidualBlock<B>,
    up3: ConvTranspose2d<B>,
    up_res3: ResidualBlock<B>,
    up4: ConvTranspose2d<B>,
    up_res4: ResidualBlock<B>,
    
    out_conv: Conv2d<B>,
    time_emb: TimeAddition<B>,
}

impl<B: Backend> Unet<B> {
    pub fn new(device: &B::Device, in_channels: usize, time_emb_dim: usize) -> Self {
        Self {
            time_emb: TimeAddition::new(time_emb_dim, device),
            
            // Downsample path with stride=2
            down1: Conv2dConfig::new([in_channels, 64], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            down_res1: ResidualBlock::new(device, 64, time_emb_dim),
            
            down2: Conv2dConfig::new([64, 128], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            down_res2: ResidualBlock::new(device, 128, time_emb_dim),
            
            down3: Conv2dConfig::new([128, 256], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            down_res3: ResidualBlock::new(device, 256, time_emb_dim),
            
            down4: Conv2dConfig::new([256, 512], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            down_res4: ResidualBlock::new(device, 512, time_emb_dim),

            // Bottleneck
            mid_res1: ResidualBlock::new(device, 512, time_emb_dim),
            mid_res2: ResidualBlock::new(device, 512, time_emb_dim),

            // Upsample path with stride=2
            up1: ConvTranspose2dConfig::new([512, 256], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            up_res1: ResidualBlock::new(device, 256, time_emb_dim),
            
            up2: ConvTranspose2dConfig::new([256, 128], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            up_res2: ResidualBlock::new(device, 128, time_emb_dim),
            
            up3: ConvTranspose2dConfig::new([128, 64], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            up_res3: ResidualBlock::new(device, 64, time_emb_dim),
            
            up4: ConvTranspose2dConfig::new([64, 64], [4, 4])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            up_res4: ResidualBlock::new(device, 64, time_emb_dim),
            
            out_conv: Conv2dConfig::new([64, in_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, time: Tensor<B, 2>) -> Tensor<B, 4> {
        let time_embed = self.time_emb.forward(time);

        // Encoder with skip connections
        let down_1 = self.down1.forward(x);
        let down_1 = self.down_res1.forward(down_1, time_embed.clone());

        let down_2 = self.down2.forward(down_1.clone());
        let down_2 = self.down_res2.forward(down_2, time_embed.clone());

        let down_3 = self.down3.forward(down_2.clone());
        let down_3 = self.down_res3.forward(down_3, time_embed.clone());

        let down_4 = self.down4.forward(down_3.clone());
        let down_4 = self.down_res4.forward(down_4, time_embed.clone());

        // Bottleneck
        let mut m = self.mid_res1.forward(down_4, time_embed.clone());
        m = self.mid_res2.forward(m, time_embed.clone());

        // Decoder with skip connections
        let u1 = self.up1.forward(m);
        let u1 = self.up_res1.forward(u1 + down_3, time_embed.clone());

        let u2 = self.up2.forward(u1);
        let u2 = self.up_res2.forward(u2 + down_2, time_embed.clone());

        let u3 = self.up3.forward(u2);
        let u3 = self.up_res3.forward(u3 + down_1, time_embed.clone());

        let u4 = self.up4.forward(u3);
        let u4 = self.up_res4.forward(u4, time_embed);

        self.out_conv.forward(u4)
    }
}

#[derive(Module, Debug)]
pub struct DiffusionModel<B: Backend> {
    unet: Unet<B>,
    num_time: usize,
}

impl<B: Backend> DiffusionModel<B> {
    pub fn new(device: &B::Device, in_channels: usize, num_time: usize) -> Self {
        Self {
            unet: Unet::new(device, in_channels, 256),
            num_time,
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

    pub fn forward(&self, x: Tensor<B, 4>, device: &B::Device) -> Tensor<B, 1> {
        let batch_size = x.dims()[0];

        // Sample random timesteps
        let t_values: Vec<i32> = (0..batch_size)
            .map(|_| (rand::random::<u32>() % self.num_time as u32) as i32)
            .collect();

        let t = Tensor::from_ints(t_values.as_slice(), device);
        
        // Add noise to images
        let (x_noisy, noise) = self.add_noise(x, t.clone(), device);

        // Get time embeddings
        let t_emb = sin_time_addition(device, t, 256);

        // Predict noise
        let noise_pred = self.unet.forward(x_noisy, t_emb);

        // MSE loss
        let noise_pred_2d: Tensor<B, 2> = noise_pred.flatten(1, 3);
        let noise_2d: Tensor<B, 2> = noise.flatten(1, 3);

        (noise_pred_2d - noise_2d).powf_scalar(2.0).mean()
    }

    pub fn sample(&self, batch_size: usize, device: &B::Device) -> Tensor<B, 4> {
        // Start with pure noise (64x64 images with 3 channels)
        let mut x: Tensor<B, 4> = Tensor::random(
            [batch_size, 3, 64, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );

        let beta = self.get_betas(device);
        let alpha_bar = self.get_alphas(device);

        // Reverse diffusion process
        for i in (0..self.num_time).rev() {
            let i_tensor = Tensor::from_ints(vec![i as i32; batch_size].as_slice(), device);
            let i_emb = sin_time_addition(device, i_tensor, 256);

            // Predict noise
            let noise_pred = self.unet.forward(x.clone(), i_emb);

            let alpha_t = alpha_bar.clone().slice([i..i + 1]);
            let beta_t = beta.clone().slice([i..i + 1]);

            let alpha_sqrt = alpha_t.clone().sqrt();
            let alpha_sqrt_inv = (Tensor::ones_like(&alpha_t) - alpha_t.clone()).sqrt();

            let [b, c, h, w] = x.dims();
            let alpha_sqrt_exp = alpha_sqrt.reshape([1, 1, 1, 1]).expand([b, c, h, w]);
            let alpha_sqrt_inv_exp = alpha_sqrt_inv.reshape([1, 1, 1, 1]).expand([b, c, h, w]);

            // Predict x0
            let pred_x0 = (x.clone() - noise_pred.clone() * alpha_sqrt_inv_exp) / alpha_sqrt_exp;

            if i > 0 {
                let alpha_prev = alpha_bar.clone().slice([i - 1..i]);
                let alpha_prev_sqrt = alpha_prev.clone().sqrt();
                let alpha_prev_inv = (Tensor::ones_like(&alpha_prev) - alpha_prev).sqrt();

                let alpha_prev_sqrt_exp = alpha_prev_sqrt.reshape([1, 1, 1, 1]).expand([b, c, h, w]);
                let alpha_prev_inv_exp = alpha_prev_inv.reshape([1, 1, 1, 1]).expand([b, c, h, w]);

                // Denoise step
                x = alpha_prev_sqrt_exp * pred_x0.clone() + alpha_prev_inv_exp * noise_pred;

                // Add noise for next step
                let noise = Tensor::random_like(&x, burn::tensor::Distribution::Normal(0.0, 1.0));
                let beta_t_sqrt_exp = beta_t.sqrt().reshape([1, 1, 1, 1]).expand([b, c, h, w]);
                x = x + beta_t_sqrt_exp * noise;
            } else {
                x = pred_x0;
            }

            if i % 50 == 0 {
                println!("  Denoising step {}/{}", self.num_time - i, self.num_time);
            }
        }

        x
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
    num_time: usize,
    device: &B::Device,
) -> Result<DiffusionModel<B>, Box<dyn std::error::Error>> {
    let recorder = CompactRecorder::new();
    let record = recorder.load(path.into(), device)?;

    let model = DiffusionModel::new(device, in_channels, num_time).load_record(record);

    Ok(model)
}

use image::imageops::FilterType;
use std::fs;
use std::path::{Path, PathBuf};

pub struct Image {
    pub size: usize,
    pub channels: usize,
    pub image_paths: Vec<PathBuf>,
    pub height: usize,
    pub width: usize,
}

impl Image {
    pub fn directory(
        dir_path: &str,
        height: usize,
        width: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
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
        })
    }

    pub fn get<B: Backend>(
        &self,
        idx: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
        if idx >= self.size {
            return Err(format!("Index {} out of bounds (size: {})", idx, self.size).into());
        }

        let img = image::open(&self.image_paths[idx])?;
        let img = img.resize_exact(self.width as u32, self.height as u32, FilterType::Lanczos3);
        let img = img.to_rgb8();

        let mut data = vec![vec![]; 3];

        for pixel in img.pixels() {
            data[0].push((pixel[0] as f32 / 255.0) * 2.0 - 1.0);
            data[1].push((pixel[1] as f32 / 255.0) * 2.0 - 1.0);
            data[2].push((pixel[2] as f32 / 255.0) * 2.0 - 1.0);
        }

        let flat_data: Vec<f32> = data.into_iter().flatten().collect();

        let tensor: Tensor<B, 4> = Tensor::<B, 1>::from_floats(flat_data.as_slice(), device)
            .reshape([1, self.channels, self.height, self.width]);
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

fn main() {
    use burn::backend::wgpu::WgpuDevice;

    println!("Initializing GPU...");
    type Backend = Autodiff<Wgpu>;
    let device = WgpuDevice::default();
    println!("Using device: {:?}", device);

    let in_channels = 3;
    let num_timesteps = 400;
    let batch_size = 8;
    let num_epochs = 50;

    let mut model = DiffusionModel::<Backend>::new(&device, in_channels, num_timesteps);
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init::<Backend, DiffusionModel<Backend>>();

    let dataset = Image::directory(r"Pictures", 64, 64).unwrap();

    println!("Starting training with {} images...", dataset.size);

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0f32;
        let num_batches = dataset.size / batch_size;

        println!("\n=== Epoch {}/{} ===", epoch + 1, num_epochs);

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(dataset.size);
            let indices: Vec<usize> = (start_idx..end_idx).collect();

            if let Ok(images) = dataset.get_batch::<Backend>(&indices, &device) {
                let loss = model.forward(images, &device);
                let grads = loss.backward();
                let grad_params = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(1e-4f64, model, grad_params);

                let loss_val = loss.into_scalar().elem::<f32>();
                total_loss += loss_val;

                if batch_idx % 5 == 0 {
                    println!("  Batch {}/{}: Loss = {:.4}", batch_idx, num_batches, loss_val);
                }
            }
        }

        let avg_loss = total_loss / num_batches as f32;
        println!("Epoch {}: Average Loss = {:.4}", epoch + 1, avg_loss);

        // Generate sample every 10 epochs
        if (epoch + 1) % 10 == 0 {
            println!("Generating sample image...");
            let generated = model.sample(1, &device);
            let generated_img: Tensor<Backend, 3> = generated.squeeze::<3>(0);

            let [_channels, height, width] = generated_img.dims();
            let data: Vec<f32> = generated_img.to_data().to_vec().unwrap();

            let img = image::RgbImage::from_fn(width as u32, height as u32, |x, y| {
                let pixel_idx = (y as usize * width) + x as usize;
                let r = ((data[pixel_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                let g = ((data[width * height + pixel_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                let b = ((data[2 * width * height + pixel_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
                image::Rgb([r, g, b])
            });
            img.save(format!("sample_epoch_{}.png", epoch + 1)).unwrap();
        }
    }

    save_model(&model, "trained_ddpm.bin").unwrap();

    println!("\n=== Generating final sample image ===");
    let generated = model.sample(1, &device);
    let generated_img: Tensor<Backend, 3> = generated.squeeze::<3>(0);

    let [_channels, height, width] = generated_img.dims();
    let data: Vec<f32> = generated_img.to_data().to_vec().unwrap();

    let img = image::RgbImage::from_fn(width as u32, height as u32, |x, y| {
        let pixel_idx = (y as usize * width) + x as usize;
        let r = ((data[pixel_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
        let g = ((data[width * height + pixel_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
        let b = ((data[2 * width * height + pixel_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8;
        image::Rgb([r, g, b])
    });
    img.save("final_generated.png").unwrap();
    println!("Final image saved to final_generated.png");
}