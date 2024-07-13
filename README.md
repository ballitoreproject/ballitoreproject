# Ballitore Project

This repository contains the code and data for the Ballitore Project, a research initiative at UCSB. The project involves various data processing, analysis, and visualization tasks centered around historical correspondence of a Quaker community.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quadrismegistus/ballitoreproject/HEAD)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Modules](#modules)
- [Notebooks](#notebooks)
- [Contributing](#contributing)
- [License](#license)

## Usage

### Option 1: [Run the code on binder](https://mybinder.org/v2/gh/quadrismegistus/ballitoreproject/HEAD)

### Option 2: Install locally

```bash
git clone https://github.com/ballitoreproject/ballitoreproject
cd ballitoreproject
pip install -r requirements.txt
```

## Project Structure

The project is organized into the following directories:

- `data`: Contains the raw and processed data files.
- `notebooks`: Contains Jupyter notebooks for data exploration, analysis, and visualization.
- `modules`: Contains Python modules for data processing, analysis, and visualization.
- `utils`: Contains utility functions for data loading, preprocessing, and visualization.

### Modules

The project includes several Python modules for data processing, analysis, and visualization. These modules are located in the `modules` directory and include:

- `data_loader`: Loads and preprocesses the data.
- `topic_modeling`: Performs topic modeling using various algorithms.
- `named_entity_recognition`: Performs named entity recognition using spaCy.
- `visualization`: Contains functions for visualizing data and results.

### Notebooks

The `notebooks` directory contains several Jupyter notebooks that demonstrate various aspects of the project. These notebooks include:

- `data_exploration`: Explores the raw data and performs initial preprocessing.
- `topic_modeling`: Demonstrates topic modeling using various algorithms.
- `named_entity_recognition`: Demonstrates named entity recognition using spaCy.
- `visualization`: Demonstrates data visualization techniques.
