use std::usize;
use burn::nn::{Linear,LinearConfig};
use burn::tensor::{Tensor,TensorData, Shape};
use burn::tensor::backend::Backend;

pub struct Pi<B: Backend>{
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B:Backend> Pi<B>{
    pub fn new(device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(1,10).init(device);
        let linear2 = LinearConfig::new(10,1).init(device);
        Self{
            linear1,
            linear2,
        }
    }
    pub fn forward(&self, input: Tensor<B,2>) ->Tensor<B,2>{
        let x = self.linear1.forward(input);
        let x = self.linear2.forward(x);
        x
    }
}

fn main(){
    type Backend = burn::backend::NdArray;
    let device = <burn::backend::NdArray as burn::prelude::Backend>::Device::default();
    let model = Pi::new(&device);

    // Create odd numbers up to 2n-1
    let n: usize = 100;
    let mut input: Vec<usize> = Vec::new();
    for i in 0..n{
        input.push(2*i+1);
    }

    // Convert to f32
    let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();

    // Create denominator tensor of shape (n,1)
    let denominator = Tensor::<Backend,2>::from_data(TensorData::new(input_f32, Shape::new([100,1])), &device);

    // Create ones tensor with shape (n,1)
    let ones = Tensor::<Backend,2>::ones([100,1],&device);

    // Divide ones by denominator
    let input1 = Tensor::<Backend,2>::div(ones,denominator);

    // Create alternating signs - FIXED: start from 0 to match input1 length
    let mut signs = Vec::new();
    for i in 0..n {  // Changed from 2..m to 0..n
        if i % 2 == 0 {
            signs.push(1.0);
        } else {
            signs.push(-1.0);
        }
    }

    // Convert signs to tensor with proper shape [n, 1]
    let signs_tensor = Tensor::<Backend,2>::from_data(
        TensorData::new(signs, Shape::new([100, 1])), 
        &device
    );

    // Multiply signs and input1
    let multiply = Tensor::<Backend,2>::mul(signs_tensor, input1.clone());

    // Sum all elements to get scalar, then multiply by 4
    let pi_approximation = multiply.sum().mul_scalar(4.0);

    // FIXED: Reshape scalar to [1, 1] for neural network input
    let input_tensor: Tensor<Backend, 2> = pi_approximation.clone().reshape([1, 1]);
    let output: Tensor<_, 2> = model.forward(input_tensor);

    // Calculate loss between output and pi value
    let abs_loss: Tensor<_, 2> = output.clone().sub_scalar(std::f32::consts::PI).abs();

    // Print results
    println!("Loss: {:?}", abs_loss);
    println!("Output: {:?}", output);
    println!("Pi approximation: {:?}", pi_approximation);
}