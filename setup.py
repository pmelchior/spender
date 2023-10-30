from setuptools import setup

long_description = open('README.md').read()

setup(
    name="spender",
    description="Spectrum encoder and decoder",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version="0.2.3",
    license="MIT",
    author="Peter Melchior",
    author_email="peter.m.melchior@gmail.com",
    url="https://github.com/pmelchior/spender",
    packages=["spender", "spender.data"],
    package_data={"skymapper.data": ['*.txt']},
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
    keywords = ['spectroscopy','autoencoder'],
    install_requires=["torch", "numpy", "accelerate", "torchinterp1d", "astropy", "humanize", "psutil", "GPUtil"]
)
