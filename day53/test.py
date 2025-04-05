from kokoro import KPipeline
from IPython.display import display, Audio
from torch.profiler import profile, record_function, ProfilerActivity
import soundfile as sf
import torch
pipeline = KPipeline(lang_code='a', device="cuda:0")
text = '''
Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, Kokoro can be deployed anywhere from production environments to personal projects.
'''
generator = pipeline(text, voice='af_heart')

# Profile the actual generation process
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
             record_shapes=True, profile_memory=True) as prof_optimized:
    for i, (gs, ps, audio) in enumerate(generator):
        sf.write(f'{i}.wav', audio, 24000)

print("\nNaive Kokoro Performance:")
print(prof_optimized.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof_optimized.export_chrome_trace("optimised_trace.json")