from setuptools import setup, find_packages

setup(
    name='structeval',
    version='0.0.2',
    author='Jean-Benoit Delbrouck',
    license='MIT',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires='>=3.10',
    install_requires=[

        'torch==2.3',
        'transformers==4.39.0',
        'radgraph',
        'rouge_score',
        'bert-score==0.3.13',
        'scikit-learn',
        'numpy<2',
        'f1chexbert'
        #'protobuf',
        #'green_score',
    ],
    packages=find_packages(),
    zip_safe=False)
