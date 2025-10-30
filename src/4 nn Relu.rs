use burn::tensor::{Tensor,TensorData,Shape};
use burn::tensor::backend::Backend;
use burn::nn::{Linear,LinearConfig,Relu,Initializer};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::module::Module;
   

#[derive(Module,Debug)]
pub struct PI<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    activation: Relu,
}

impl<B:Backend> PI<B>{
        pub fn new(device: &B::Device) ->Self{
            let linear1 = LinearConfig::new(1, 64)
                .with_initializer(Initializer::KaimingNormal { gain: (1.0), fan_out_only: false })
                .init(device);
            let linear2 = LinearConfig::new(64,32)
                .with_initializer(Initializer::KaimingNormal { gain: (1.0), fan_out_only: (false)})
                .init(device);
            let linear3 = LinearConfig::new(32,16)
                .with_initializer(Initializer::KaimingNormal {gain:(1.0),fan_out_only:(false)})
                .init(device);
            let linear4 = LinearConfig::new(16,1)
                .with_initializer(Initializer::KaimingNormal {gain: (1.0),fan_out_only: (false)})
                .init(device);
            let activation = Relu;
            Self{
                linear1,
                linear2,
                linear3,
                linear4,
                activation,
            }
        }

        pub fn forward(&self, input: Tensor<B,2>) ->Tensor<B,2>{
            let x = self.activation.forward(self.linear1.forward(input));
            let x = self.activation.forward(self.linear2.forward(x));
            let x = self.activation.forward(self.linear3.forward(x));
            let x = self.linear4.forward(x);
            x
            
        }
}

fn main(){
    type Backend = burn_autodiff::Autodiff<burn::backend::NdArray>;

    let device = <Backend as burn::prelude::Backend>::Device::default();
    let mut  model = PI::new(&device);
    let mut optimizer = AdamConfig::new().init::<Backend, PI<Backend>>();

     let n: usize = 5000;
    let mut input: Vec<usize> = Vec::new();
    for i in 0..n{
        input.push(2*i+1);
    }

    // converting the vector to a tensor of shape (n,1)

    let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();

    // creating a tensor of shape (n,1) from the input vector

    let denominator = Tensor::<Backend,2>::from_data(TensorData::new(input_f32, Shape::new([5000,1])), &device);
    
    // creating a tensor of ones with shape (n,1)
    let ones = Tensor::<Backend,2>::ones([5000,1],&device);

    // dividing ones by the denominator tensor
    let input1 = Tensor::<Backend,2>::div(ones,denominator);

    // To create alternate 1 and -1 signs
    let mut signs = Vec::new();
    let m: usize = n;
    for i in 0..m {
        if i % 2 == 0 {
            signs.push(1.0);
        } else {
            signs.push(-1.0);
        }
    }

    // convert signs Vec<f32> to a tensor
   let signs_tensor = Tensor::<Backend,2>::from_data(
        TensorData::new(signs, Shape::new([5000, 1])), 
        &device
    );

    // multipying sign tensor and input 1
    let multiply = Tensor::<Backend,2>::mul(signs_tensor, input1.clone());
    
    // definfing an output tensor

    let pi_approximation = multiply.sum().mul_scalar(4.0);

    // Use input1 directly since it's not used after this point
    // Reshape _output_scalar to a 2D tensor of shape [1, 1]

    let input_tensor: Tensor<Backend, 2> = pi_approximation.reshape([1,1]);
    let output: Tensor<_, 2> = model.forward(input_tensor.clone());

    // calculate the original pi value
    
    let pi_value = Tensor::<Backend,2>::from_data(
        TensorData::new(vec![std::f32::consts::PI], Shape::new([1,1])),
        &device
    );

    // calculating the loss between output and pi value

    let _pi_approximation_tensor = output.clone().reshape([1,1]).detach();
    let target_tensor = Tensor::<Backend,2>::from_floats([[std::f32::consts::PI]], &device);

    let epochs = 100;
        for epoch in 0..epochs {

        let output = model.forward(input_tensor.clone());

        let loss = burn::nn::loss::MseLoss::new().forward(output.clone(), target_tensor.clone(), burn::nn::loss::Reduction::Mean);

        let grads = loss.backward();
        let grad_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(1e-4, model, grad_params);

         // Print loss every 1 epochs

        if epoch % 1 == 0{
            println!("Epoch: {}, Loss: {:?}", epoch, loss);
        }
    }

    let final_output = model.forward(input_tensor.clone());
    let final_loss = burn::nn::loss::MseLoss::new().forward(
        final_output.clone(), 
        target_tensor, 
        burn::nn::loss::Reduction::Mean
    );

     // printing the output tensor
    println!("\n Model output: {:?}",final_output.clone());

    println!("\n Pi value: {:?}",pi_value.clone()); 

    println!("\n Final Loss: {:?}",final_loss.clone());

    let final_output_scalar: f32 = final_output.clone().into_scalar();
    let pi_value_scalar: f32 = pi_value.clone().into_scalar();

// Call plot function
    
}


   

