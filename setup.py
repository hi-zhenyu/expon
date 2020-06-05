import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="expon", # Replace with your own username
    version="0.0.5",
    author="Zhenyu Huang",
    author_email="zyhuang.gm@gmail.com",
    description="Experiment tool for deep learning (PyTorch).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hi-zhenyu/expon",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'markdown',
          'numpy',
          'matplotlib',
    ],
    python_requires='>=3.6',
    include_package_data=True,
)

# python3 setup.py sdist bdist_wheel
# python3 -m twine upload --repository testpypi dist/*
# python3 -m twine upload dist/*