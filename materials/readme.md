# NexusFlow 

This is a minimal, standalone PyTorch implementation of the **NexusFlow** module, as presented in our paper on partially supervised learning for disparate tasks.

The core idea is to bridge the gap between different tasks (e.g., dense mapping + sparse tracking) when you don't have all the labels for all your data. NexusFlow is a lightweight, plug-and-play module that aligns the feature distributions of these tasks, forcing the main model to learn a more unified representation.

## How It Works

The magic is in the **Distribution Alignment Loss**. We don't touch your main model's architecture. Instead, we add a small, parallel module that operates "off to the side":

```
                                  |--> Task Decoder 1 (e.g., Mapping) -> Loss_1
                                  |
[Frozen BEV Encoder] -> [Shared BEV Feature] ->|--> Task Decoder 2 (e.g., Tracking) -> Loss_2
                                  |
                                  |--> [NexusFlow Module] -> Alignment_Loss
```

The NexusFlow module itself consists of:
* A pair of **Surrogate Networks** that take the shared BEV feature as input.
* Each surrogate network uses a **Deformable Attention** layer to efficiently squash the large feature map into a small vector.
* This vector is then passed through an **Invertible Coupling Layer** (from normalizing flows) to transform it into a canonical space.
* Finally, we calculate the **MSE loss** between the two outputs of the surrogate networks. This loss is the signal that tells the model to align the distributions.

## Integrating NexusFlow into Your Model

This module is designed to be easily dropped into existing BEV-based perception models. The process is simple:

#### 1. Add the Module
Instantiate the `NexusFlow` module alongside your main model.

#### 2. Plumb it into your Training Loop
Inside your training loop, after you get the shared BEV feature, simply pass it through the module to get the alignment loss. Then add this loss to your main task loss.

Here's a pseudo-code example:

```python
# Assuming `model` is your main network and `nexusflow_module` is an instance of our class.
# ... inside your training loop ...

# 1. Get the shared feature from your frozen encoder
bev_feature = model.bev_encoder(x) # Shape: (B, C, H, W)

# 2. (Optional) Fine-tune your decoders with the original task losses
map_output = model.map_decoder(bev_feature)
track_output = model.track_decoder(bev_feature)
l_task = calculate_partial_loss(map_output, track_output, labels, mask)

# 3. Get the alignment loss from NexusFlow
z_map, z_track = nexusflow_module(bev_feature)
l_align = mse_loss(z_map, z_track)

# 4. Combine the losses and backpropagate
# lambda_align is your hyperparameter to balance the loss terms
total_loss = l_task + lambda_align * l_align

total_loss.backward()
optimizer.step()
```

As our paper shows, you can either add `l_align` from the start of training or, for slightly better performance, pre-train your model first and then add `l_align` during a fine-tuning phase.

## Setup

The code requires PyTorch and the `deformable-attention-pytorch` library.

```bash
pip install torch
pip install deformable-attention-pytorch
```

## Running the code

To see the module in action, just run the script:

```bash
python NexusFlow.py
```

The script will build the module, create a random BEV feature tensor, pass it through, and compute the alignment loss, showing that the forward and backward passes work correctly.

