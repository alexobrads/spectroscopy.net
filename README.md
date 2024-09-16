# Automated Spectral Emission Line Mapping

This project aims to automate the process of mapping measured emission lines from pixel positions to wavelengths in spectroscopy, which is typically done manually by scientists during spectroscopic analysis.
By automating the emission line mapping process, we significantly reduce the time scientists spend on manual identification and improve the accuracy and reliability of their measurements. 
This README provides an overview and detailed math breakdown of the models and methods used in the project.

**Please see project_notes notebook for a detailed breakdown method and its application**

## Overview

### Problem Statement

When observing the spectra of stars, scientists measure emission lines at various pixel positions. These need to be mapped to known wavelengths. Traditionally, this is achieved manually by identifying common wavelength patterns and using them as reference points. This project automates that process by creating a database of wavelength-pair features and using a likelihood function to determine the accuracy of the found solution.

### Project Goal

To develop a method that:
1. Generates a store of wavelength-pair features.
2. Uses these features to map observed pixel positions to known wavelengths.
3. Evaluates the accuracy of the mapping using a likelihood function.

## Methodology

### KD-Trees

KD-Trees are used to efficiently find the nearest neighbor in k-dimensional space. In this project we use KD-Trees to store wavelength pairs:

1. **Create the Memmaps**: Create memmaps for both indices and ratios for our reference list for quick retrieval and file handling.

    ```python
    import numpy as np

    ratios = np.memmap('ratios.dat', dtype='float32', mode='w+', shape=(N_ref, N_ref))
    indices = np.memmap('indices.dat', dtype='int32', mode='w+', shape=(N_ref, N_ref))
    ```

2. **Populate the memmaps**: Loop through the reference data to populate the memmaps with all possible combinations of wavelength ratios.

    ```python
    for i in range(N_ref):
        for j in range(N_ref):
            if i != j:
                ratios[i, j] = wl_ref[i] / wl_ref[j]
                indices[i, j] = j
    ```

3. **Build the KD-Tree**: Build a KD-Tree on this memmap for efficient querying.

    ```python
    from scipy.spatial import KDTree

    tree = KDTree(ratios)
    ```

### Mathematical Models

#### Background Model

The background model considers the probability that any emission line in our camera occurs at any position within the range of the reference data. This acts as a benchmark for comparison.

```python
def log_background_probability(N_obs, ref_max, ref_min):
    return np.sum([
        -1 * np.log(ref_max - ref_min)
        for _ in range(N_obs)
    ])
```

#### Foreground Model

This model calculates the probability of our hypothesized solution. For computational efficiency, we use the logarithmic form:

```python
def log_likelihood(obs, ref, sigma):
    chi_squared = np.sum((obs - ref) ** 2 / (2 * sigma ** 2))
    K = -1 * np.log(np.sqrt(2 * np.pi * sigma ** 2))  # constant term
    return K - 0.5 * chi_squared
```

We aim to minimize \(\chi^{2}\) to maximize the likelihood.

#### Mixture Model

Given that not all observed lines are in the reference list, we account for outliers with a mixture model:

```python
def mixture_model_likelihood(N, fg_likelihood, bg_likelihood, q):
    return np.prod([
        (fg_likelihood[i] ** q[i]) * (bg_likelihood[i] ** (1 - q[i]))
        for i in range(N)
    ])
```

### Optimization

To optimize the solution, we use Stan, a platform for statistical modeling and high-performance computing. The optimization seeks to fit the best possible model to the data by evaluating different parameters and minimizing the \(\chi^{2}\) value.

## Next Steps

1. Implement the KD-Tree structure and population.
2. Code the likelihood functions for both the background and foreground models.
3. Develop the optimization routine leveraging Stan.
4. Validate the approach with a set of test data and compare it against manual mappings.

