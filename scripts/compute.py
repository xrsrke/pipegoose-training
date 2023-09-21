# Define function to calculate memory, FLOPs, and training time
def calculate_training_metrics(tensor_parallel_size, data_parallel_size, pipeline_parallel_size,
                               num_layers, num_params, num_heads, hidden_size, throughput_gpu,
                               context_length, batch_size, num_epochs):
    # Constants
    bytes_per_param_fp32 = 4  # fp32 uses 4 bytes per parameter

    num_gpus = tensor_parallel_size*data_parallel_size*pipeline_parallel_size

    # Memory Calculations
    ## Model Memory
    model_memory = num_params * bytes_per_param_fp32
    ## Optimizer Memory (Vanilla AdamW uses 12 bytes per parameter)
    optimizer_memory = 12 * num_params
    ## Activation Memory
    ### Using equation for memory_activations^Selective Recomputation
    activation_memory = context_length * batch_size * hidden_size * num_layers * (10 + 24 / tensor_parallel_size) * \
    (5*((num_heads*context_length)/(hidden_size*tensor_parallel_size)))
    ## Gradient Memory (Stored in fp32)
    gradient_memory = num_params * bytes_per_param_fp32
    ## Total Training Memory (3D-parallelism with ZeRO-1)
    total_memory = (model_memory / (pipeline_parallel_size * tensor_parallel_size)) + \
                   (optimizer_memory / num_gpus) + \
                   (activation_memory / tensor_parallel_size) + \
                   (gradient_memory / pipeline_parallel_size)

    # FLOPs Calculations
    ## Using C = tau * T => T = C / tau
    ### Forward Pass FLOPs
    forward_flops = 2 * num_params * (context_length * batch_size)
    ### Backward Pass FLOPs
    backward_flops = 4 * num_params * (context_length * batch_size)
    ## Total FLOPs per Epoch
    total_flops_per_epoch = (forward_flops + backward_flops)
    ## Total Training Time per Epoch in seconds
    training_time_per_epoch = total_flops_per_epoch / (throughput_gpu * 1e12)
    ## Total Training Time for all Epochs in seconds
    total_training_time = training_time_per_epoch * num_epochs

    # Print Results
    print(f"\nResults:")
    print(f"Model Memory: {model_memory / 1e9} GB")
    print(f"Optimizer Memory: {optimizer_memory / 1e9} GB")
    print(f"Activation Memory: {activation_memory / 1e9} GB")
    print(f"Gradient Memory: {gradient_memory / 1e9} GB")
    print(f"Total Training Memory: {total_memory / 1e9} GB")
    print(f"Total FLOPs per Epoch: {total_flops_per_epoch:.2e}")
    print(f"Training Time per Epoch: {training_time_per_epoch:.2f} seconds")
    print(f"Total Training Time: {total_training_time:.2f} seconds")


if __name__ == "__main__":
    calculate_training_metrics(
        tensor_parallel_size=2,
        data_parallel_size=2,
        pipeline_parallel_size=2,
        num_layers=12,
        num_params=125e9,
        num_heads=12,
        hidden_size=768,
        throughput_gpu=120,
        context_length=512,
        batch_size=9765,
        num_epochs=25
    )