from setuptools import setup, find_packages

setup(
    name='bert-model-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A BERT-based model for natural language processing tasks.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',  # or 'tensorflow' depending on your implementation
        'numpy',
        'pandas',
        'scikit-learn',
        'transformers',  # if using Hugging Face's Transformers library
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)