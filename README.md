# Computational Diffraction Simulation  
**Numerical Diffraction Patterns via Deterministic Integration and Monte Carlo Methods**

## Overview

This project implements a computational physics simulation of optical diffraction patterns using numerical methods. The goal is to reproduce and analyse diffraction intensity distributions by directly evaluating the diffraction integral using both deterministic numerical integration and Monte Carlo sampling techniques.

The work compares accuracy, convergence behaviour, and computational efficiency across methods, and demonstrates how stochastic sampling can be used to approximate wave interference patterns.

This project was developed as part of a computational physics coursework exercise.

---

## Physics Background

Diffraction patterns arise from interference of waves passing through an aperture. In scalar diffraction theory, the observed intensity can be computed from an integral over the aperture:

\[
I(x,y) \propto \left| \int_{\text{aperture}} e^{ik r} \, dA \right|^2
\]

where contributions from each aperture point are summed with appropriate phase.

This project evaluates this integral numerically using:

- Grid-based numerical integration
- Monte Carlo area sampling
- Statistical convergence analysis

---

## Methods Implemented

### Deterministic Numerical Integration
- Discretised aperture grid
- Direct summation of phase contributions
- Controlled resolution studies
- Baseline accuracy reference

### Monte Carlo Diffraction Estimator
- Random sampling over aperture region
- Stochastic phase summation
- Variance and convergence analysis
- Scaling behaviour with sample size

### Comparative Analysis
- Convergence vs sample count
- Variance behaviour
- Accuracy vs compute cost
- Visual pattern comparison

---
## Structure
src/ — simulation code  
plots/ — generated figures  
report/ — written analysis  

## Author
Sammy Bouchebaba — Physics BSc (University of Bristol)


## Repository Structure

