import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qcfpga",
    version="0.0.4",
    author="Emmanuel KIEFFER",
    author_email="Emmanuel KIEFFER",
    description="An FPGA-OpenCL based quantum computer simulator based on Adam's Kelly QCGPU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ekieffer/qcfpga",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    setup_requires=['pytest-runner'],
    install_requires=['mako', 'pyopencl', 'pybind11', 'numpy'],
    tests_require=["pytest"]
)
