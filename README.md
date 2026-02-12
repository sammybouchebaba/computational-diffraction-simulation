# Computational Diffraction Simulation

**Numerical diffraction patterns via deterministic integration and Monte Carlo methods**

## Overview

This project implements a computational physics simulation of optical diffraction patterns by directly evaluating diffraction integrals over an aperture using numerical methods.

Both deterministic quadrature and Monte Carlo sampling approaches are implemented and compared. The code generates 1D and 2D diffraction intensity distributions for rectangular and circular apertures in both Fresnel (near-field) and Fraunhofer (far-field) regimes.

The project investigates:

- Direct numerical evaluation of diffraction integrals
- Monte Carlo stochastic estimators
- Convergence and variance behaviour
- Accuracy vs computational cost tradeoffs

Developed as part of a computational physics coursework project.

---

## Physics Background

In scalar diffraction theory, the complex field at a screen point is obtained by summing wave contributions from all points in the aperture with the appropriate phase.

Numerically, this is computed as an integral of a complex exponential phase factor over the aperture area. The observed intensity is the squared magnitude of the resulting complex field.

Two phase models are implemented:

**Fresnel diffraction (near field):**
phase proportional to ((x − x')² + (y − y')²) / (2z)

**Fraunhofer diffraction (far field):**
phase proportional to (x'x + y'y) / z

The simulation evaluates these integrals directly using numerical integration and Monte Carlo sampling.

---

## Methods Implemented

### Deterministic Numerical Integration (SciPy dblquad)

- Direct evaluation of real and imaginary parts of the diffraction integral
- Uses SciPy `dblquad` adaptive quadrature
- Applied to:
  - 1D slit diffraction
  - 2D rectangular apertures
  - 2D circular apertures
- Used as an accuracy reference method
- Includes quadrature error estimates (1D case)

---

### Monte Carlo Diffraction Estimator

- Uniform random sampling inside circular apertures
- Phase contributions averaged stochastically
- Demonstrates:
  - Variance scaling proportional to 1/sqrt(N)
  - Cost vs accuracy tradeoffs
  - Practical stochastic integration of oscillatory integrals

---

### Comparative Performance Study

Includes a performance comparison module measuring:

- Monte Carlo runtime vs sample count
- Numerical integration runtime vs grid resolution
- Error scaling behaviour
- Efficiency metric defined as roughly 1 / (error × time)

Results are plotted on log–log axes.

---

## Features

### 1D Diffraction

- Fresnel and Fraunhofer regimes
- Relative intensity plots
- Absolute and relative error estimates
- Custom aperture and screen distance

---

### 2D Rectangular Aperture Diffraction

- Fresnel and Fraunhofer modes
- Adjustable aperture width and height
- Adjustable resolution and viewing window

---

### 2D Circular Aperture Diffraction

- Deterministic integration
- Fresnel and Fraunhofer phase models
- Circular aperture boundary handling

---

### Monte Carlo Circular Diffraction

- Stochastic sampling inside aperture
- Adjustable sample count
- Resolution control
- Visual comparison vs deterministic integration

