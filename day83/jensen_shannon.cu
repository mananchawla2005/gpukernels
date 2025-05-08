__global__ void jsd_kernel(
    const float *__restrict__ logits,  // input in logspace, logits = log Q
    const float *__restrict__ targets, // ground truth in logspace, targets = log P
    float *__restrict__ loss_output,
    float *__restrict__ gradients,
    const int *__restrict__ sample_labels,
    float mixture_weight,
    int num_valid_samples,
    int ignored_label_value,
    int dimension_size)
{
    int sample_idx = blockIdx.x;                             // Each block processes one row
    int feature_idx = threadIdx.x + blockDim.x * blockIdx.y; // Columns are processed by threads

    if (feature_idx >= dimension_size)
        return;

    int flat_idx = sample_idx * dimension_size + feature_idx;
    float logit_val = logits[flat_idx];
    float target_val = targets[flat_idx];
    float gradient_val = 0.0f;
    float sample_loss = 0.0f;

    if (sample_labels && sample_labels[sample_idx] == ignored_label_value)
    {
        gradients[flat_idx] = 0.0f;
        return;
    }

    if (mixture_weight == 0.0f)
    { // Forward KL
        float target_prob = expf(target_val);
        sample_loss = target_prob * (target_val - logit_val);
        gradient_val = -target_prob;
    }
    else if (mixture_weight == 1.0f)
    { // Reverse KL
        float logit_prob = expf(logit_val);
        sample_loss = logit_prob * (logit_val - target_val);
        gradient_val = sample_loss + logit_prob;
    }
    else
    { // JSD or Generalized KL
        float query_prob = expf(logit_val);
        float prior_prob = expf(target_val);
        float mixture_prob = mixture_weight * prior_prob + (1 - mixture_weight) * query_prob;
        float log_mixture_prob = logf(mixture_prob);
        sample_loss = mixture_weight * prior_prob * target_val + (1 - mixture_weight) * query_prob * logit_val - mixture_prob * log_mixture_prob;
        gradient_val = (1 - mixture_weight) * query_prob * (logit_val - log_mixture_prob);
    }

    float normalization_factor = 1.0f / num_valid_samples;
    loss_output[flat_idx] = sample_loss * normalization_factor;
    gradients[flat_idx] = gradient_val * normalization_factor;
}

void launch_jsd_kernel(
    const float *logits,
    const float *targets,
    float *loss_output,
    float *gradients,
    const int *sample_labels,
    float mixture_weight,
    int batch_size,
    int num_valid_samples,
    int ignored_label_value,
    int dimension_size,
    cudaStream_t stream = nullptr)
{
    // Choose appropriate block and grid dimensions
    const int threads_per_block = 256;

    // Calculate grid dimensions
    // - x dimension: one block per sample
    // - y dimension: enough blocks to cover all features
    dim3 block_dim(threads_per_block);
    dim3 grid_dim(
        batch_size,
        (dimension_size + threads_per_block - 1) / threads_per_block);

    // Launch kernel
    jsd_kernel<<<grid_dim, block_dim, 0, stream>>>(
        logits,
        targets,
        loss_output,
        gradients,
        sample_labels,
        mixture_weight,
        num_valid_samples,
        ignored_label_value,
        dimension_size);
}