#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:13:16 2024

@author: sammybouchebaba
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
import random
import time

# Constants
wavelength = 589e-9  # Wavelength of light (m)
k = 2 * np.pi / wavelength  # Wavenumber
E0 = 1.0  # Electric field constant

# Fresnel diffraction kernels
def fresnel_real(yp, xp, y, x, k, z):
    return np.cos(k / (2 * z) * ((x - xp)**2 + (y - yp)**2))

def fresnel_imag(yp, xp, y, x, k, z):
    return np.sin(k / (2 * z) * ((x - xp)**2 + (y - yp)**2))

# Input Handling Functions
def get_valid_input(prompt, min_value=None, max_value=None, default=None):
    """
    Prompt the user for input with validation.
    """
    while True:
        try:
            user_input = input(f"{prompt} (Default: {default}): ").strip()
            if user_input == "" and default is not None:
                return default
            value = float(user_input)
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                print(f"Please enter a value between {min_value} and {max_value}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


# Part 1: 1D Fresnel and Fraunhofer Diffraction with Error Analysis
def part1_1d_diffraction_with_error(x_vals, y=0, z=0.05, aperture_width=2e-5):
    """
    Computes 1D Fresnel or Fraunhofer diffraction intensities with error estimates.
    """
    xp1, xp2 = -aperture_width / 2, aperture_width / 2  # Aperture limits
    intensities = []
    absolute_errors = []
    relative_errors = []

    C = k / (2 * np.pi * z)  # Constant factor

    for x in x_vals:
        # Compute real and imaginary parts using dblquad
        real_part, real_error = dblquad(fresnel_real, xp1, xp2, lambda _: xp1, lambda _: xp2, args=(y, x, k, z))
        imag_part, imag_error = dblquad(fresnel_imag, xp1, xp2, lambda _: xp1, lambda _: xp2, args=(y, x, k, z))

        # Compute intensity and absolute error
        E = C * (real_part + 1j * imag_part)
        intensity = np.abs(E)**2
        intensities.append(intensity)

        intensity_error = (real_error**2 + imag_error**2)**0.5 * C
        absolute_errors.append(intensity_error)

        # Compute relative error
        relative_error = intensity_error / intensity if intensity > 0 else 0
        relative_errors.append(relative_error)

    return np.array(intensities), np.array(absolute_errors), np.array(relative_errors)

def plot_1d_diffraction(mode, z=None, aperture_width=None, x_range=None, error_analysis=False):
    """
    Plots 1D Fresnel or Fraunhofer diffraction patterns with optional error analysis.
    """
    if mode == 'fraunhofer':
        z, aperture_width = 0.05, 2e-5
        x_vals = np.linspace(-0.008, 0.008, 200)
    elif mode == 'fresnel':
        z, aperture_width = 0.2, 1.9e-3
        x_vals = np.linspace(-0.0015, 0.0015, 200)
    elif mode == 'custom':
        z = z or get_valid_input("Enter custom z value (m)", min_value=0.001, max_value=1, default=0.2)
        aperture_width = aperture_width or get_valid_input("Enter custom aperture width (m)", min_value=1e-6, max_value=1e-2, default=1.1e-3)
        if x_range is None:  # Only ask for x_range if it is not already provided
            x_range = get_valid_input("Enter x-range for the plot (half-width in meters)", min_value=1e-4, max_value=1, default=0.002)
        x_vals = np.linspace(-x_range, x_range, 200)
    else:
        raise ValueError("Invalid mode. Please choose 'fraunhofer', 'fresnel', or 'custom'.")

    if error_analysis:
        # Perform error analysis
        intensities, abs_errors, rel_errors = part1_1d_diffraction_with_error(x_vals, z=z, aperture_width=aperture_width)
        rel_intensities = intensities / np.max(intensities)

        # Plot relative intensity
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(x_vals, rel_intensities, label="Relative Intensity", color="blue")
        plt.xlabel("Screen Coordinate (m)")
        plt.ylabel("Relative Intensity")
        plt.title(f"1D Diffraction Intensity ({mode.capitalize()} Mode)")
        plt.grid()
        plt.legend()

        # Plot error analysis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(x_vals, rel_errors, label="Relative Error", color="pink")
        ax1.set_xlabel("Screen Coordinate (m)")
        ax1.set_ylabel("Relative Error", color="pink")
        ax1.tick_params(axis="y", labelcolor="pink")
        ax1.set_title(f"Error Analysis for 1D Diffraction ({mode.capitalize()} Mode)")

        ax2 = ax1.twinx()
        ax2.plot(x_vals, abs_errors, label="Absolute Error", color="blue", alpha=0.5)
        ax2.set_ylabel("Absolute Error", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax1.grid()

        plt.tight_layout()
        plt.show()
    else:
        # Standard intensity plot
        intensities, _, _ = part1_1d_diffraction_with_error(x_vals, z=z, aperture_width=aperture_width)
        rel_intensities = intensities / np.max(intensities)
        plt.plot(x_vals, rel_intensities, label="Relative Intensity", color="blue")
        plt.xlabel("Screen Coordinate (m)")
        plt.ylabel("Relative Intensity")
        plt.title(f"1D Diffraction Intensity ({mode.capitalize()} Mode)\n z = {z:.3f} m, Aperture Width = {aperture_width:.3e} m")
        plt.grid()
        plt.legend()
        plt.show()



# Part 2: 2D Rectangular Diffraction with Enhanced Customisation
def part2_2d_diffraction(mode, z=None, aperture_width=None, aperture_height=None, x_range=None, resolution=None):
    """
    Simulates 2D diffraction for rectangular or square apertures with customizable x-range and resolution.

    Parameters:
    - mode: 'fraunhofer', 'fresnel', or 'custom'
    - z: Distance from aperture to screen (m)
    - aperture_width: Width of the aperture (m)
    - aperture_height: Height of the aperture (m)
    - x_range: Half-width of the x-range (m) for the plot
    - resolution: Number of points for x and y coordinates
    """
    if mode == 'fraunhofer':
        z = 0.005
        aperture_width = 2e-6
        aperture_height = aperture_width  # Square aperture for Fraunhofer
        x_range = x_range or 0.005  # Default range
        resolution = resolution or 80
    elif mode == 'fresnel':
        z = 0.2
        aperture_width = 1.1e-3
        aperture_height = aperture_width  # Square aperture for Fresnel
        x_range = x_range or 0.002  # Default range
        resolution = resolution or 60
    elif mode == 'custom':
        if z is None or aperture_width is None or aperture_height is None:
            raise ValueError("For custom mode, provide z, aperture_width, and aperture_height.")
        x_range = x_range or 0.005  # Default custom range
        resolution = resolution or 60
    else:
        raise ValueError("Invalid mode. Choose 'fraunhofer', 'fresnel', or 'custom'.")

    # Define x and y coordinate ranges
    x_vals = np.linspace(-x_range, x_range, resolution)
    y_vals = np.linspace(-x_range, x_range, resolution)

    # Initialize intensity matrix
    intensities = np.zeros((len(y_vals), len(x_vals)))

    # Compute intensity using dblquad for a rectangular aperture
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            real_part, _ = dblquad(
                fresnel_real,
                -aperture_width / 2, aperture_width / 2,
                lambda _: -aperture_height / 2, lambda _: aperture_height / 2,
                args=(y, x, k, z)
            )
            imag_part, _ = dblquad(
                fresnel_imag,
                -aperture_width / 2, aperture_width / 2,
                lambda _: -aperture_height / 2, lambda _: aperture_height / 2,
                args=(y, x, k, z)
            )
            E = (k * E0 / (2 * np.pi * z)) * (real_part + 1j * imag_part)
            intensities[j, i] = np.abs(E)**2

    # Normalize the intensity
    normalized_intensity = intensities / np.max(intensities)

    # Plot the normalized intensity
    plt.imshow(
        normalized_intensity,
        extent=[-x_range, x_range, -x_range, x_range],
        cmap='nipy_spectral_r', origin='lower'
    )
    plt.colorbar(label="Relative Intensity")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(
        f"2D Rectangular Diffraction ({mode.capitalize()} Mode)\n"
        f"z = {z:.3f} m, Aperture Width = {aperture_width:.3e} m, "
        f"Aperture Height = {aperture_height:.3e} m, Resolution = {resolution} x {resolution}"
    )
    plt.show()


# Part 3: 2D Circular Diffraction
def fresnel_circular_phase(xp, yp, x, y, z):
    return k / (2 * z) * ((x - xp)**2 + (y - yp)**2)

def fraunhofer_circular_phase(xp, yp, x, y, z):
    return k * (xp * x + yp * y) / z

def part3_2d_circular_diffraction(mode='fresnel', z=None, aperture_radius=None, resolution=100, x_range=None, y_range=None):
    """
    Simulate 2D Circular Diffraction for Fresnel and Fraunhofer modes with customizability.

    Parameters:
    - mode: 'fresnel' or 'fraunhofer'
    - z: Screen distance
    - aperture_radius: Radius of the aperture
    - resolution: Number of points for x and y coordinates
    - x_range: Half-width of the range for x-coordinates
    - y_range: Half-width of the range for y-coordinates
    """
    # Default parameters for predefined modes
    if mode == 'fraunhofer':
        z = 0.005
        aperture_radius = 2e-6 / 2
        x_range = x_range or 0.004  # Default x-range for Fraunhofer
        y_range = y_range or 0.004  # Default y-range for Fraunhofer
    elif mode == 'fresnel':
        z = 0.2
        aperture_radius = 1.9e-3 / 2
        x_range = x_range or 0.0015  # Default zoomed-in x-range for Fresnel
        y_range = y_range or 0.0015  # Default zoomed-in y-range for Fresnel
    elif mode == 'custom':
        if z is None or aperture_radius is None or x_range is None or y_range is None:
            raise ValueError("For custom mode, provide z, aperture_radius, x_range, and y_range.")
    else:
        raise ValueError("Invalid mode. Choose 'fresnel', 'fraunhofer', or 'custom'.")

    # Generate x and y values
    x_vals = np.linspace(-x_range, x_range, resolution)
    y_vals = np.linspace(-y_range, y_range, resolution)

    # Initialize intensity matrix
    intensities = np.zeros((len(y_vals), len(x_vals)))

    def yp1_func(xp):
        return -np.sqrt(aperture_radius**2 - xp**2)

    def yp2_func(xp):
        return np.sqrt(aperture_radius**2 - xp**2)

    # Calculate intensities
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            try:
                phase_func = (
                    fresnel_circular_phase if mode == 'fresnel' else fraunhofer_circular_phase
                )
                real_part, _ = dblquad(
                    lambda yp, xp: np.cos(phase_func(xp, yp, x, y, z)),
                    -aperture_radius, aperture_radius,
                    lambda xp: yp1_func(xp), lambda xp: yp2_func(xp)
                )
                imag_part, _ = dblquad(
                    lambda yp, xp: np.sin(phase_func(xp, yp, x, y, z)),
                    -aperture_radius, aperture_radius,
                    lambda xp: yp1_func(xp), lambda xp: yp2_func(xp)
                )
                E = (k * E0 / (2 * np.pi * z)) * (real_part + 1j * imag_part)
                intensities[j, i] = np.abs(E)**2
            except ValueError:
                intensities[j, i] = 0.0

    # Normalize intensities
    intensities /= np.max(intensities)

    # Plot
    plt.imshow(intensities, extent=[-x_range, x_range, -y_range, y_range],
               cmap='nipy_spectral_r', origin='lower')
    plt.colorbar(label="Relative Intensity")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(
        f"2D Circular Diffraction ({mode.capitalize()} Mode)\n"
        f"z = {z:.3f} m, Aperture Radius = {aperture_radius*2:.3e} m"
    )
    plt.show()



# Part 4: Monte Carlo Circular Diffraction
def monte_carlo_circular_diffraction(mode='fresnel', N=3000, z=None, aperture_radius=None, x_range=None, y_range=None, resolution=None):
    """
    Simulate Monte Carlo Circular Diffraction with customizability.

    Parameters:
    - mode: 'fresnel', 'fraunhofer', or 'custom'
    - N: Number of Monte Carlo samples
    - z: Distance to screen
    - aperture_radius: Radius of the circular aperture
    - x_range: Half-width of the range for x-coordinates
    - y_range: Half-width of the range for y-coordinates
    - resolution: Number of points in each dimension
    """
    # Default parameters for predefined modes
    if mode == 'fraunhofer':
        z = 0.005
        aperture_radius = 2e-6 / 2
        x_range = x_range or 0.0045
        y_range = y_range or 0.0045
        resolution = resolution or 90
    elif mode == 'fresnel':
        z = 0.2
        aperture_radius = 1.1e-3 / 2
        x_range = x_range or 0.0013
        y_range = y_range or 0.0013
        resolution = resolution or 110
    elif mode == 'custom':
        if z is None or aperture_radius is None or x_range is None or y_range is None or resolution is None:
            raise ValueError("For custom mode, provide z, aperture_radius, x_range, y_range, and resolution.")
    else:
        raise ValueError("Invalid mode. Choose 'fresnel', 'fraunhofer', or 'custom'.")

    # Generate x and y values
    x_vals = np.linspace(-x_range, x_range, resolution)
    y_vals = np.linspace(-y_range, y_range, resolution)

    # Initialize intensity matrix
    intensities = np.zeros((len(y_vals), len(x_vals)))

    # Compute Monte Carlo samples
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            samples = []
            for _ in range(N):
                xp = random.uniform(-aperture_radius, aperture_radius)
                yp = random.uniform(-aperture_radius, aperture_radius)
                if xp**2 + yp**2 <= aperture_radius**2:
                    phase = (
                        fresnel_circular_phase(xp, yp, x, y, z)
                        if mode == 'fresnel' else fraunhofer_circular_phase(xp, yp, x, y, z)
                    )
                    samples.append(np.exp(1j * phase))
            if samples:
                E = (k * E0 / (2 * np.pi * z)) * np.mean(samples)
                intensities[j, i] = np.abs(E)**2

    # Normalize intensities
    intensities /= np.max(intensities)

    # Plot
    plt.imshow(intensities, extent=[-x_range, x_range, -y_range, y_range],
               cmap='nipy_spectral_r', origin='lower')
    plt.colorbar(label="Relative Intensity")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(
        f"Monte Carlo Circular Diffraction ({mode.capitalize()} Mode)\n"
        f"z = {z:.3f} m, Aperture Radius = {aperture_radius*2:.3e} m"
    )
    plt.show()

    
# Part 5 Monte Carlo vs dblquad efficiency comparison
def comprehensive_method_comparison():
    """
    Evaluate performance metrics for Monte Carlo and numerical integration approaches.
    """
    diffraction_type = input("Enter type ('fresnel' or 'fraunhofer'): ").strip().lower()

    # Configure parameters based on diffraction type
    if diffraction_type == 'fraunhofer':
        distance = 0.2
        slit_width = 3.46e-5
        monte_carlo_iterations = [100000, 200000, 500000, 1000000, 1500000]
        numerical_resolutions = [30, 40, 50, 60, 70, 80, 90]
    elif diffraction_type == 'fresnel':
        distance = 0.2
        slit_width = 1.09e-3
        monte_carlo_iterations = [250000, 500000, 750000, 1000000, 1500000, 2000000, 3000000]
        numerical_resolutions = [2, 4, 5, 10, 15]
    slit_radius = slit_width / 2

    monte_carlo_performance = []
    numerical_performance = []
    monte_carlo_times = []
    numerical_times = []

    min_val = 1e-9  # Minimum value for division protection

    print("\nInitiating comprehensive performance comparison...")

    # Monte Carlo Method Analysis
    print("Evaluating Monte Carlo method performance...")
    for iterations in monte_carlo_iterations:
        timer_start = time.time()
        for _ in range(iterations):
            pos_x = random.uniform(-slit_radius, slit_radius)
            pos_y = random.uniform(-slit_radius, slit_radius)
            if pos_x**2 + pos_y**2 <= slit_radius**2:
                wave_phase = k / (2 * distance) * ((0 - pos_x)**2 + (0 - pos_y)**2)
                _ = np.exp(1j * wave_phase)  # Calculate wave function
        elapsed = time.time() - timer_start

        # Calculate stochastic error estimate
        monte_carlo_error = 1 / np.sqrt(iterations)
        monte_carlo_performance.append(1 / (monte_carlo_error * max(elapsed, min_val)))
        monte_carlo_times.append(elapsed)

    # Numerical Integration Analysis
    print("Evaluating numerical integration performance...")
    for resolution in numerical_resolutions:
        coord_x = np.linspace(-0.01, 0.01, resolution)
        coord_y = np.linspace(-0.01, 0.01, resolution)
        timer_start = time.time()
        for j, y_val in enumerate(coord_y):
            for i, x_val in enumerate(coord_x):
                real_component, _ = dblquad(fresnel_real, -slit_radius, slit_radius,
                                       lambda x: -np.sqrt(slit_radius**2 - x**2),
                                       lambda x: np.sqrt(slit_radius**2 - x**2),
                                       args=(y_val, x_val, k, distance))
        elapsed = time.time() - timer_start

        # Set numerical integration error estimate
        numerical_error = 1e-3  # Fixed error assumption
        numerical_performance.append(1 / (numerical_error * max(elapsed, min_val)))
        numerical_times.append(elapsed)

    # Visualization
    print("Generating performance visualization...")
    plt.figure(figsize=(10, 6))

    # Plot stochastic results
    plt.plot(monte_carlo_times, monte_carlo_performance, label="Monte Carlo Method",
             marker="o", linestyle='-', color="blue")

    # Plot numerical results
    plt.plot(numerical_times, numerical_performance, label="Numerical Integration",
             marker="x", linestyle='-', color="orange")

    # Configure plot
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Computation Time (seconds, log scale)")
    plt.ylabel("Method Efficiency (1 / Error × Time, log scale)")
    plt.title(f"Comprehensive Performance Analysis ({diffraction_type.capitalize()})")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()



# Menu System
def menu():
    while True:
        print("\nFresnel Diffraction Simulation")
        print("1. 1D Diffraction Pattern")
        print("2. 2D Rectangular Diffraction")
        print("3. 2D Circular Diffraction")
        print("4. Monte Carlo Circular Diffraction")
        print("5. Efficiency Comparison (Monte Carlo vs dblquad)")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ").strip().lower()

        if choice == '1':
            print("\n1D Diffraction Options:")
            print("1. Standard Intensity Plot (Fraunhofer)")
            print("2. Standard Intensity Plot (Fresnel)")
            print("3. Custom Intensity Plot")
            print("4. Error Analysis")
            
            sub_choice = input("Select an option (1-4): ").strip()
            
            if sub_choice == '1':
                plot_1d_diffraction('fraunhofer')
            elif sub_choice == '2':
                plot_1d_diffraction('fresnel')
            elif sub_choice == '3':
                try:
                    z = float(input("Enter custom z value (m): "))
                    aperture_width = float(input("Enter custom aperture width (m): "))
                    x_range = float(input("Enter x-range for the plot (half-width in meters): "))
                    plot_1d_diffraction('custom', z=z, aperture_width=aperture_width, x_range=x_range)
                except ValueError:
                    print("Invalid input! Returning to main menu.")
            elif sub_choice == '4':
                mode = input("Enter mode for error analysis ('fresnel' or 'fraunhofer'): ").strip().lower()
                plot_1d_diffraction(mode, error_analysis=True)
            else:
                print("Invalid sub-choice. Returning to main menu.")


        elif choice == '2':
            print("\n2D Rectangular Diffraction Options:")
            print("1. Fraunhofer: Default Parameters")
            print("2. Fresnel: Default Parameters")
            print("3. Custom: Input your own values.")
            
            sub_choice = input("Select an option (1-3): ").strip()
            
            if sub_choice == '1':
                part2_2d_diffraction('fraunhofer')
            elif sub_choice == '2':
                part2_2d_diffraction('fresnel')
            elif sub_choice == '3':
                try:
                    z = float(input("Enter custom z value (m): "))
                    aperture_width = float(input("Enter custom aperture width (m): "))
                    aperture_height = float(input("Enter custom aperture height (m): "))
                    x_range = float(input("Enter custom x-range half-width (m) for the plot: "))
                    resolution = int(input("Enter custom resolution (number of points): "))
                    part2_2d_diffraction(
                        mode='custom',
                        z=z,
                        aperture_width=aperture_width,
                        aperture_height=aperture_height,
                        x_range=x_range,
                        resolution=resolution
                    )
                except ValueError:
                    print("Invalid input! Returning to main menu.")
            else:
                print("Invalid sub-choice. Returning to main menu.")

        elif choice == '3':
            print("\n2D Circular Diffraction Options:")
            print("1. Fraunhofer: z = 0.005 m, Aperture Radius = 2e-6 m")
            print("2. Fresnel: z = 0.2 m, Aperture Radius = 1.9e-3 m")
            print("3. Custom: Input your own values.")
            
            sub_choice = input("Select an option (1-3): ").strip()
            
            if sub_choice == '1':
                part3_2d_circular_diffraction('fraunhofer')
            elif sub_choice == '2':
                part3_2d_circular_diffraction('fresnel')
            elif sub_choice == '3':
                try:
                    z = float(input("Enter custom z value (m): "))
                    aperture_radius = float(input("Enter custom aperture radius (m): "))
                    x_range = float(input("Enter custom x-range half-width (m): "))
                    y_range = float(input("Enter custom y-range half-height (m): "))
                    resolution = int(input("Enter custom resolution (number of points): "))
                    part3_2d_circular_diffraction(
                        mode='custom',
                        z=z,
                        aperture_radius=aperture_radius,
                        x_range=x_range,
                        y_range=y_range,
                        resolution=resolution
                    )
                except ValueError:
                    print("Invalid input! Returning to main menu.")
            else:
                print("Invalid sub-choice. Returning to main menu.")

        elif choice == '4':
            print("\nMonte Carlo Circular Diffraction Options:")
            print("1. Fraunhofer: z = 0.005 m, Aperture Radius = 2e-6 m")
            print("2. Fresnel: z = 0.2 m, Aperture Radius = 1.1e-3 m")
            print("3. Custom: Input your own values.")
            
            sub_choice = input("Select an option (1-3): ").strip()
            
            if sub_choice == '1':
                monte_carlo_circular_diffraction('fraunhofer')
            elif sub_choice == '2':
                monte_carlo_circular_diffraction('fresnel')
            elif sub_choice == '3':
                try:
                    z = float(input("Enter custom z value (m): "))
                    aperture_radius = float(input("Enter custom aperture radius (m): "))
                    x_range = float(input("Enter custom x-range half-width (m): "))
                    y_range = float(input("Enter custom y-range half-width (m): "))
                    resolution = int(input("Enter custom resolution (number of points): "))
                    N = int(input("Enter number of Monte Carlo samples (e.g., 3000): "))
                    monte_carlo_circular_diffraction(
                        mode='custom',
                        z=z,
                        aperture_radius=aperture_radius,
                        x_range=x_range,
                        y_range=y_range,
                        resolution=resolution,
                        N=N
                    )
                except ValueError:
                    print("Invalid input! Returning to main menu.")
            else:
                print("Invalid sub-choice. Returning to main menu.")

        elif choice == '5':
         comprehensive_method_comparison()

        elif choice == '6':
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    menu()


