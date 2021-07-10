
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="okex-py-v5",
    version="1.0",
    author="newusually",
    author_email="493076373@qq.com",
    description="OKEX python sdk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantmew/okex-py",
    project_urls={
        "Bug Tracker": "https://github.com/okex-py-v5",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.7'
)