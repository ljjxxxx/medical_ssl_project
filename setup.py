from setuptools import setup, find_packages

setup(
    name="chest-xray-classification",
    version="1.0.0",
    packages=find_packages(),
    description="Chest X-ray Disease Classification using Deep Learning",
    author="Jiaxiang Li",
    author_email="2633575992@qq.com",
    install_requires=[
        "torch==2.0.1",
        "torchvision==0.15.2",
        "torchaudio==2.0.2",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "Pillow==10.0.1",
        "matplotlib==3.7.3",
        "scikit-learn==1.3.2",
        "tqdm==4.66.1",
        "gradio==4.44.1",
        "urllib3==2.0.7",
    ],
)