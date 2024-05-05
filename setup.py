import setuptools

with open("README.md", "r",encoding="utf-8") as fh:
    long_description = fh.read()

# Setting up
setuptools.setup(
    name="scientistshiny",
    version="0.0.2",
    author="Duverier DJIFACK ZEBAZE",
    author_email="duverierdjifack@gmail.com",
    description="Python library to easily improve multivariate Exploratory Data Analysis graphs",
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=["shiny>=0.9.0",
                      "numpy>=1.26.4",
                      "matplotlib>=3.8.4",
                      "scikit-learn>=1.2.2",
                      "pandas>=2.2.2",
                      "numexpr>=2.10.0",
                      "plotnine>=0.10.1",
                      "scientisttools>=0.1.5"],
    python_requires=">=3.10",
    include_package_data=True,
    package_data={"": ["data/*.xlsx",
                       "data/*.xls",
                       "data/*.txt",
                       "data/*.csv",
                       "data/*.rda"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)