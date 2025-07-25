from setuptools import setup,find_packages


setup(
    name='vlm_grpo',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
        "torchaudio==2.6.0+cu124",
        "transformers==4.52.4",
        "unsloth==2025.6.12",
        "unsloth_zoo==2025.6.8",
        "bitsandbytes",
        "accelerate",
        "xformers==0.0.29.post3",
        "triton==3.2.0",
        "cut_cross_entropy",
        "peft==0.15.2",
        "trl==0.18.2",
        "sentencepiece",
        "protobuf",
        "datasets",
        "huggingface_hub",
        "hf_transfer",
        "safetensors",
        "regex",
        "tokenizers"
    ],
)