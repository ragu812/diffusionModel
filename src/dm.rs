use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig,Linear, LinearConfig,  PaddingConfig2d};
use burn::recorder::{Recorder,CompactRecorder};
//Defining a encoder circuit in a VAE using CNN
//Using BatchNorm to normalize the output and improve the performance
//using PaddingConfig2d for 

#[derive(Module,Debug)]
pub struct SimpleEncoder<B:Backend>{
    conv1: Conv2d<B>,
    bn_1: BatchNorm<B,2>,
    conv2: Conv2d<B>,
    bn_2: BatchNorm<B,2>,
    conv3: Conv2d<B>,
    bn_3: BatchNorm<B,2>,
    conv4: Conv2d<B>,
    bn_4: BatchNorm<B,2>,
    mean: Linear<B>,
    variance: Linear<B>,
    latent_dimen: usize,
}

//The encoder maps the input to the latent space which compresses the input data into lower dimensions.
//It uses kernal function to analyse the spatial hierarchies in the input data

impl<B:Backend> SimpleEncoder<B>{
    pub fn new(in_channels:usize,latent_dimen:usize,device: &B::Device)-> Self{
        Self{
             conv1: Conv2dConfig::new([in_channels,64],[3,3])
                .with_stride([1,1])
                .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            bn_1: BatchNormConfig::new(64).init(device),


            conv2: Conv2dConfig::new([64,128],[3,3])
                .with_stride([1,1])
                .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            bn_2: BatchNormConfig::new(128).init(device),


            conv3: Conv2dConfig::new([128,256],[3,3])
                .with_stride([1,1])
                .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            bn_3: BatchNormConfig::new(256).init(device),


            conv4: Conv2dConfig::new([256,512],[3,3])
                .with_stride([1,1])
                .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            bn_4: BatchNormConfig::new(512).init(device),

            //Using Linear Layer to conver the input from spatial into latent space
            //Linear Layer will compress the input into fixed size vector
            mean: LinearConfig::new(512*4*4,latent_dimen).init(device),


            variance: LinearConfig::new(512*4*4,latent_dimen).init(device),
            latent_dimen
        }
    }

    pub fn forward(&self, input1:Tensor<B,4>)->(Tensor<B,2>,Tensor<B,2>){
        let x = self.bn_1.forward(self.conv1.forward(input1));
        let x = activation::relu(x);

        let x = self.bn_2.forward(self.conv2.forward(x));
        let x = activation::relu(x);
        
        let x = self.bn_3.forward(self.conv3.forward(x));
        let x = activation::relu(x);
        
        let x = self.bn_4.forward(self.conv4.forward(x));
        let x = activation::relu(x);

        let x = x.flatten(1,2);

        let mean = self.mean.forward(x.clone());
        let variance = self.variance.forward(x.clone());

        (mean,variance)
    }
}

//Building a Decoder using CNN and Linear Layers
//Latent Space doesnt have a space just a !D vector

#[derive(Module,Debug)]
pub struct SimpleDecoder<B:Backend>{
    transfer: Linear<B>,
    reverse1: ConvTranspose2d<B>,
    bn_1: BatchNorm<B,2>,
    reverse2: ConvTranspose2d<B>,
    bn_2: BatchNorm<B,2>,
    reverse3: ConvTranspose2d<B>,
    bn_3: BatchNorm<B,2>,
    reverse4: ConvTranspose2d<B>,
    latent_dimen: usize,
    output: usize,
} 

impl<B:Backend> SimpleDecoder<B>{
    pub fn new(output:usize,latent_dimen:usize,device: &B::Device)-> Self{
        Self{
            transfer: LinearConfig::new(latent_dimen,512*4*4).init(device),

            reverse1: ConvTranspose2dConfig::new([512,256],[3,3])
                .with_stride([1,1])
                .with_padding([1,1])
                .init(device),
            bn_1: BatchNormConfig::new(256).init(device),

            reverse2: ConvTranspose2dConfig::new([256,128],[3,3])
                .with_stride([1,1])
                .with_padding([1,1])
                .init(device),
            bn_2: BatchNormConfig::new(128).init(device),

            reverse3: ConvTranspose2dConfig::new([128,64],[3,3])
                .with_stride([1,1])
                .with_padding([1,1])
                .init(device),
            bn_3: BatchNormConfig::new(64).init(device),

            reverse4: ConvTranspose2dConfig::new([64,output],[3,3])
                .with_stride([1,1])
                .with_padding([1,1])
                .init(device),

            latent_dimen,
            output,
        }
    }

    pub fn forward(&self, input2:Tensor<B,2>) -> Tensor<B,4>{

        let y = self.transfer.forward(input2);
        let y = activation::relu(y);
        let y = y.reshape([y.dims()[0],512,4,4]);

        let y = self.reverse1.forward(y);
        let y= self.bn_1.forward(y);
        let y = activation::relu(y); 

        let y = self.reverse2.forward(y);
        let y= self.bn_2.forward(y);
        let y = activation::relu(y);

        let y = self.reverse3.forward(y);
        let y= self.bn_3.forward(y);
        let y = activation::relu(y);

        let y = self.reverse4.forward(y);
        let y= activation::tanh(y);
    }
}

//Combining both Encoder and Decoder to form a VAE
#[derive(Module,Debug)]
pub struct Vae<B:Backend>{
    encoder: SimpleEncoder<B>,
    decoder: SimpleDecoder<B>,
}

impl<B:Backend> Vae<B>{
    pub fn new(in_channels:usize,latent_dimen:usize, device: &B::Device)-> Self{
        Self{
            encoder: SimpleEncoder::new(in_channels,latent_dimen,device),
            decoder: SimpleDecoder::new(latent_dimen,in_channels,device),
        }
    }

    pub fn parametrize(&self, mean: Tensor<B,2> , variance: Tensor<B,2>) -> Tensor<B,2>{
        let std = (variance.clone() * 0.5).exp();
        let noise = Tensor::random_like(&std, burn::tensor::Distribution::Normal(0.0,1.0));
        mean + noise * std
    }

    pub fn forward(&self, x:Tensor<B,4>) -> (Tensor<B,4>, Tensor<B, 2>, Tensor<B, 2>) {
    // Single encoder outputs combined features
        let (mean, variance) = self.encoder.forward(x);
        let z = self.parameterize(mean.clone(), variance.clone());
        let reconstruction = self.decoder.forward(z);
        (reconstruction, mean, variance)
    }


    pub fn encode(&self, x:Tensor<B,4>)-> Tensor<B,2>{
        let (mean, variance) = self.encoder.forward(x);
        self.parametrize(mean,variance)
    }
    pub fn decode(&self, y:Tensor<B,2>)->Tensor<B,4>{

        self.decoder.forward(y)
    } 


//calculating th VAE loss using KL Loss and Reconstruction Loss

    pub fn vae_loss(mean: Tensor<B,2> , variance:Tensor<B,2>, reconstruction:Tensor<B,2>,input:Tensor<B,2>)-> Tensor<B,1>{
        let kl_loss= -0.5 * (Tensor::ones_like(&variance) + variance.clone().log() - mean.powf_scalar(2.0) - variance.exp()).sum();
        let recon_loss = burn::nn::loss::MseLoss::new().forward(reconstruction,input,burn::nn::loss::Reduction::Mean);
        recon_loss + kl_loss
    }
}
//Now we have finished the architecture of VAE

//Now we will build the Diffusion Model using UNet and Time Embedding

#[derive(Module,Debug)]
pub struct TimeAddition<B:Backend>{
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B:Backend> TimeAddition<B>{
    pub fn new(dim:usize, device: &B::Device)-> Self{
       Self{ 
                linear1 : LinearConfig::new(dim,dim*4).init(device),
                linear2 : LinearConfig::new(dim*4,dim).init(device),
       }
    }
    pub fn forward(&self, t: Tensor<B,2>)-> Tensor<B,2>{
        let z = self.linear1.forward(t);
        let z = activation::relu(z);
        self.linear2.forward(z);
    }
}

pub fn sin_time_addition<B: Backend>(device: &B::Device, time: Tensor<B,1,Int>, dim: usize)-> Tensor<B,2>{
   
    let batch_size = time.dims()[0];
    let half_dim =dim/2;
    let t = time.float();
    let frequencies:Vec<f32> = (0..half_dim)
        .map(|i|{
            let exp = (i as f32)*4.0*std::f32::consts::LN_10/(half_dim as f32);
            exp.exp()
        }).collect();
    let freqs = Tensor::from_floats(frequencies.as_slice(), device)
        .reshape([1,half_dim])
        .repeat_dim( 0,batch_size);
    
    let arg = t.reshape([batch_size,1]).repeat_dim(1, half_dim)*freqs;

    let sin_emb = arg.clone().sin();
    let cos_emb = arg.cos();
    
    Tensor::cat(vec![sin_emb, cos_emb], 1)
}

#[derive(Module,Debug)]
pub struct ResiduelBlock<B:Backend>{
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    bn1: BatchNorm<B,2>,
    bn2: BatchNorm<B,2>,
    time: Linear<B>,
}

impl<B:Backend> ResiduelBlock<B>{
    pub fn new(device: &B::Device, channels:usize,time_dim: usize)-> Self{
        Self{
            conv1: Conv2dConfig::new([channels,channels],[3,3])
                .with_stride([1,1])
                .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            bn1: BatchNormConfig::new(channels).init(device),

            conv2: Conv2dConfig::new([channels,channels], [3,3])
                .with_stride([1,1])
                .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            bn2: BatchNormConfig::new(channels).init(device),

            time: LinearConfig::new(time_dim,channels).init(device),
        }
    }

    pub fn forward(&self, x:Tensor<B,4>, t_emb:Tensor<B,2>)-> Tensor<B,4>{

        let h = self.conv1.forward(x.clone());
        let h = self.bn1.forward(h);
        let h = activation::silu(h);

        let t = self.time.forward(t_emb);
        let t = activation::silu(t);
        let t = t.reshape([t.dims()[0], t.dims()[1], 1, 1]);
        let h = h+t;

        let h = self.conv2.forward(h);
        let h = self.bn2.forward(h);
        let h = activation::silu(h+x);

    }
}

#[derive(Module,Debug)]
pub struct Unet<B:Backend>{
    down1: Conv2d<B>,
    down_res1: ResiduelBlock<B>,
    down2: Conv2d<B>,
    down_res2: ResiduelBlock<B>,
    down3: Conv2d<B>,
    down_res3: ResiduelBlock<B>,
    down4: Conv2d<B>,
    down_res4: ResiduelBlock<B>,
    down5: Conv2d<B>,
    down_res5: ResiduelBlock<B>,
    mid_res1:ResiduelBlock<B>,
    mid_res2: ResiduelBlock<B>,
    mid_res3: ResiduelBlock<B>,
    up1: ConvTranspose2d<B>,
    up_res1: ResiduelBlock<B>,
    up2: ConvTranspose2d<B>,
    up_res2: ResiduelBlock<B>,
    up3: ConvTranspose2d<B>,
    up_res3: ResiduelBlock<B>,
    up4: ConvTranspose2d<B>,
    up_res4: ResiduelBlock<B>,
    out_conv: Conv2d<B>,
    time_emd: TimeAddition<B>,
}

impl<B:Backend> Unet<B>{
    pub fn new(device: &B::Device, in_channels: usize, time_emd: usize)-> Self{
        Self{

            time_emd: TimeAddition::new(time_emd,device),
            down1: Conv2dConfig::new([in_channels,64],[3,3])
                .with_stride([2,2])
                .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            down_res1: ResiduelBlock::new(device,64,time_emd),
            down2: Conv2dConfig::new([64,128],[3,3])
                .with_stride([2,2])
                .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            down_res2: ResiduelBlock::new(device,128,time_emd),
            down3: Conv2dConfig::new([128,256],[3,3])
                .with_stride([2,2])
                .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            down_res3: ResiduelBlock::new(device,256,time_emd),
            down4: Conv2dConfig::new([256,512],[3,3])
                .with_stride([2,2])
                .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            down_res4: ResiduelBlock::new(device,512,time_emd),
            down5: Conv2dConfig::new([512,1024],[3,3])
                .with_stride([2,2])
                .with_padding(PaddingConfig2d::Explicit(1,1))
                .init(device),
            down_res5: ResiduelBlock::new(device,1024,time_emd),

            mid_res1: ResiduelBlock::new(device,1024,time_emd),
            mid_res2: ResiduelBlock::new(device,1024,time_emd),
            mid_res3: ResiduelBlock::new(device,1024,time_emd),

            up1: ConvTranspose2dConfig::new([1024,512],[3,3])
                .with_stride([2,2])
                .with_padding([1,1])
                .init(device),
            up_res1: ResiduelBlock::new(device,512,time_emd),
            up2: ConvTranspose2dConfig::new([512,256],[3,3])
                .with_stride([2,2])
                .with_padding([1,1])
                .init(device),
            up_res2: ResiduelBlock::new(device,256,time_emd),
            up3: ConvTranspose2dConfig::new([256,128],[3,3])
                .with_stride([2,2])
                .with_padding([1,1])
                .init(device),
            up_res3: ResiduelBlock::new(device,128,time_emd),
            up4: ConvTranspose2dConfig::new([128,64],[3,3])
                .with_stride([2,2])
                .with_padding([1,1])
                .init(device),
            up_res4: ResiduelBlock::new(device,64,time_emd),
            out_conv: Conv2dConfig::new([64,in_channels],[3,3]).init(device),

        }
    }

    pub fn forward(&self, x:Tensor<B,4>, time: Tensor<B,2>)-> Tensor<B,4>{

        let time_embed = self.time_emd.forward(time);

        let down_1 = self.down1.forward(x);
        let down_1 = self.down_res1.forward(down_1,time_embed.clone());
        
        let down_2 = self.down2.forward(down_1.clone());
        let down_2 = self.down_res2.forward(down_2,time_embed.clone());

        let down_3 = self.down3.forward(down_2.clone());
        let down_3 = self.down_res3.forward(down_3,time_embed.clone());

        let down_4 = self.down4.forward(down_3.clone());
        let down_4 = self.down_res4.forward(down_4,time_embed.clone());

        let down_5 = self.down5.forward(down_4.clone());
        let down_5 = self.down_res5.forward(down_5,time_embed.clone());

        let mut m = self.mid_res1.forward(down_5,time_embed.clone());
        m = self.mid_res2.forward(m, time_embed.clone());
        m = self.mid_res3.forward(m, time_embed.clone());

        let u1 = self.up1.forward(m);
        let u1 =  self.up_res1.forward(u1+down_3, time_embed.clone());

        let u2 = self.up2.forward(u1.clone());
        let u2 = self.up_res2.forward(u2+down_2, time_embed.clone());

        let u3 = self.up3.forward(u2.clone());
        let u3 = self.up_res3.forward(u3+down_1, time_embed.clone());

        let u4 = self.up4.forward(u3.clone());
        let u4 = self.up_res4.forward(u4, time_embed.clone());

        self.out_conv.forward(u4)

    }
}

pub struct DiffusionModel<B:Backend>{
    unet: Unet<B>,
    vae: Vae<B>,
    num_time: usize,  //Tells the number of steps used to denoise it
    latent_dimen: usize,  //The dimension size of the latent space
}

impl<B:Backend> DiffusionModel<B>{
    pub fn new(device: &B::Device, latent_dimen: usize, in_channels: usize, num_time: usize)-> Self{

        Self{
            vae: Vae::new(in_channels,latent_dimen,device),
            unet: Unet::new(device,latent_dimen,num_time),
            num_time,
            latent_dimen
        }
    }

    //We use betas because it defines the amount of noise which is added to the image and num_time gives the number of steps that this noise is added

    pub fn get_betas(&self, device: &B::Device)-> Tensor<B,1>{
        let beta_start = 0.000001;
        let beta_end = 0.5;

        let steps = self.num_time;

        let betas: Vec<f32>= (0..steps)
            .map(|i|{
                beta_start + (beta_start - beta_end) * (i as f32) / ((steps-1) as f32)
            }).collect();

        Tensor::from_floats(betas.as_slice(),device)
    }

    pub fn get_alphas(&self, device: &B::Device) -> Tensor<B, 1> {
        let betas = self.get_betas(device);
        let alphas = Tensor::ones_like(&betas) - betas;
        
        // Cumulative product
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

    pub fn add_noise(&self,x: Tensor<B,2> , time: Tensor<B,1>, device: &B::Device) -> (Tensor<B,2> , Tensor<B,2>){
        let alpha = self.get_alphas(device);
        let noise = Tensor::random_like(&x , burn::tensor::Distribution::Normal(0.0, 1.0));

        let batch_size = x.dims()[0];
        let latent_dimen = x.dims()[1];
        
        let mut noisy_samples = Vec::new();

    for i in 0..batch_size {
    
        let time_t = time.clone().slice([i..i+1]).into_scalar().elem::<i32>() as usize;
        
        
        let alpha_t = alpha.clone().slice([time_t..time_t+1]);
        
        let sqrt_alpha = alpha_t.clone().sqrt();
        let sqrt_one_minus_alpha = (Tensor::ones_like(&alpha_t) - alpha_t).sqrt();
        
        let xi = x.clone().slice([i..i+1]);
        let noise_i = noise.clone().slice([i..i+1]);

        let noisy_i = xi * sqrt_alpha.repeat_dim(0, latent_dimen) 
                    + noise_i * sqrt_one_minus_alpha.repeat_dim(0, latent_dimen);

            let noisy = Tensor::cat(
                noisy_samples, 0);
    }

    pub fn noise_predict(&self, time_emb: Tensor<B,1,Int>, x_t: Tensor<B,4>)-> Tensor<B,4>{
        self.unet.forward(x_t, time_emb)
    }

    pub fn forward(&self, x: Tensor<B,4>, device: &B::Device)-> Tensor<B,1>{


        let batch_size = x.dim()[0];
        let z = self.encode(x.clone());

        let t_values: Vec<i32> = (0..batch_size)
            .map(|_| (use burn::rand::random::<u32>() % self.num_time as u32) as i32).collect();

        let t = Tensor::from_ints(t_values.as_slice(), device);
        let (z_noisy, noise) = self.add_noise(z, t.clone());

        let laten_dimen = z_noisy.dims()[1];
        let z_4d = z_noisy.reshape([batch_size,self.latent_dimen,1,1]);

        let t_emb = sin_time_addition(device, t, 256);

        let noise_pred = self.predict_noise(z_4d, t_emb);
        let noise_pred_2d = noise_pred.flatten(1,3);

        let noise_2d = noise.reshape([batch_size, self.latent_dimen]);

        (noise_pred_2d - noise_pred).powf_scalar(2.0).mean()
    }

    pub fn sample(&self, latent_dimen:usize, batch_size: usize, device: &B::Device)-> Tensor<B,4>{

        let mut z = Tensor::random([batch_size,latent_dimen],burn::tensor::Distribution::Normal(0.0,1.0), device);

        let beta = self.get_betas(device);
        let alpha_1 = self.get_alphas(device);
        for i in(0..batch_size).rev(){
            let i_tensor = Tensor::from_ints(&vec![i as i32;batch_size], device);
            let i_emb = sin_time_addition(device, i_tensor, 256);

            let z_4d = z.clone().reshape([batch_size,latent_dimen,1,1]);
            let noise_pred = self.noise_predict(z_4d,i_emb);
            let noise_pred_2d = noise_pred.flatten(1,3);

            let alpha_t = alpha_1.clone().slice([i..i+1]);
            let beta_t = beta.clone().slice([i..i+1]);

            let alpha_sqrt = alpha_t.sqrt();
            let alpha_sqrt_1 = (Tensor::ones_like(&alpha_t))-alpha_sqrt;

            let pred_x0 = (z.clone() - noise_pred_2d.clone() *alpha_sqrt_1(0, latent_dimen)) 
            /alpha_sqrt(0, latent_dimen);

            if i > 0{
                let alpha_prev = alpha_1.clone().slice([i-1..i]);
                let alpha_perv_sqrt = alpha_prev.sqrt();
                let alpha_perv_1 = (Tensor::ones_like(&alpha_prev) - alpha_prev).sqrt();

                z = alpha_perv_sqrt.repeat(0,latent_dimen)*pred_x0
                    +alpha_perv_1.repeat(0,latent_dimen)*noise_pred_2d;
                
                let noise = Tensor::random_like(&z, burn::tensor::Distribution::Normal(0.0, 1.0));
                     z = z + beta_t.sqrt().repeat(0, latent_dimen) * noise;
            } else {
                 z = pred_x0;
            }

         }
         self.decode(z)
    }

}

//In this we are going to save this entire model and using it for future reference


pub fn save_model<B: Backend>(model: &DiffusionModel<B>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let recorder = CompactRecorder::new();
    recorder.record(model.clone().into_record(), path.into())?;
    println!("Model saved to {}", path);
    Ok(())
}

pub fn load_model<B:Backend>(path: &str,in_channels: usize,latent_dimen: usize,num_time: usize,device: &B::Device) -> Result<DiffusionModel<B>, Box<dyn std::error::Error()>>{

    let recorder = CompactReader::new();
    let record = recorder.load(path.into(), device)?;

    let model = DiffusionModel::new(in_channels, latent_dimen,num_time,device)
        .load_record(record);

    save_model(&model, "trained.bin")?;
}

use std::fs;
use std::path::{Path,PathBuf};
use image::{DynamicImage,GenericImageView, imageops::FilterType};

pub struct Image{
    pub size: usize,
    pub channels: usize,
    pub image_paths: Vec<PathBuf>,
    pub height: usize,
    pub width: usize,
}

impl Image{
    pub fn directory(dir_path: &str, height: usize, width: usize) 
    -> Result<Self, Box<dyn std::error::Error>>{
        let path = Path::new(dir_path);

        if !path.exists() {
            return Err(format!("Directory does not exist: {}", dir_path).into());
        }
        let mut image_paths = Vec::new();

        for e in fs::read_dir(path)?{

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
            channels: 3, // RGB
            height,
            width,
        })
}
 pub fn from_paths(paths: Vec<String>, height: usize, width: usize) -> Result<Self, Box<dyn std::error::Error>> {
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
        })
    }
    
    /// Load a single image and convert to tensor
    pub fn get<B: Backend>(&self, idx: usize, device: &B::Device) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
        if idx >= self.size {
            return Err(format!("Index {} out of bounds (size: {})", idx, self.size).into());
        }
        
        // Load and preprocess image
        let img = image::open(&self.image_paths[idx])?;
        let img = img.resize_exact(self.width as u32, self.height as u32, FilterType::Lanczos3);
        let img = img.to_rgb8();
        
        // Convert to float and normalize to [-1, 1]
        let mut data = Vec::with_capacity(self.channels * self.height * self.width);
        
        for pixel in img.pixels() {
            // Normalize from [0, 255] to [-1, 1]
            data.push((pixel[0] as f32 / 255.0) * 2.0 - 1.0); // R
            data.push((pixel[1] as f32 / 255.0) * 2.0 - 1.0); // G
            data.push((pixel[2] as f32 / 255.0) * 2.0 - 1.0); // B
        }
        
        Ok(Tensor::from_floats(data.as_slice(), device)
            .reshape([1, self.channels, self.height, self.width]))
    }
    
    /// Get batch of images
    pub fn get_batch<B: Backend>(
        &self, 
        indices: &[usize], 
        device: &B::Device
    ) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
        let mut batch_images = Vec::new();
        
        for &idx in indices {
            batch_images.push(self.get::<B>(idx, device)?);
        }
        
        Ok(Tensor::cat(batch_images, 0))
    }
}
}

// ============================================================================
// Training Loop Example
// ============================================================================

pub fn train_ldm_epoch<B: Backend, O: Optimizer<LatentDiffusionModel<B>, B>>(
    model: &LatentDiffusionModel<B>,
    dataset: &SimpleImageDataset,
    optimizer: &mut O,
    device: &B::Device,
    batch_size: usize,
) -> f32 {
    let mut total_loss = 0.0;
    let num_batches = dataset.size / batch_size;
    
    for batch_idx in 0..num_batches {
        let mut batch_images = Vec::new();
        
        for i in 0..batch_size {
            let idx = batch_idx * batch_size + i;
            batch_images.push(dataset.get::<B>(idx, device));
        }
        
        let images = Tensor::cat(batch_images, 0);
        let loss = model.forward(images, device);
        
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, model);
        *model = optimizer.step(1e-4, model.clone(), grads);
        
        total_loss += loss.into_scalar();
    }
    
    total_loss / num_batches as f32

fn main() {
    use burn::backend::Wgpu;
    
    type Backend = Wgpu;
    let device = Default::default();
    
    // Hyperparameters
    let in_channels = 3;
    let latent_dim = 256;
    let time_dim = 256;
    let num_timesteps = 1000;
    let batch_size = 4;
    let num_epochs = 10;
    
    // Create model
    let ldm = LatentDiffusionModel::<Backend>::new(
        in_channels,
        latent_dim,
        time_dim,
        num_timesteps,
        &device,
    );
    
    // Create dataset
    let dataset = SimpleImageDataset::new_random(100, 3, 64, 64);
    
    // Create optimizer
    let mut optim = AdamConfig::new().init();
    
    // Training loop
    println!("Starting training...");
    for epoch in 0..num_epochs {
        let loss = train_ldm_epoch(&ldm, &dataset, &mut optim, &device, batch_size);
        println!("Epoch {}: Loss = {:.4}", epoch + 1, loss);
    }
}
