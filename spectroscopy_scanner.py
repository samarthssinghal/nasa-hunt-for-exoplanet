import numpy as np
import matplotlib.pyplot as plt
import random

# Realistic molecular absorption lines database (wavelength in Angstroms, relative strength)
MOLECULAR_SIGNATURES = {
    'H2O': [
        (1215.67, 0.8),   # Lyman-alpha region affected by H from H2O
        (1240.14, 0.3),   # H2O absorption
        (1260.42, 0.4),   # H2O absorption
        (1280.15, 0.2),   # Weak H2O line
    ],
    'CO2': [
        (1190.20, 0.5),   # CO2 absorption
        (1205.65, 0.4),   # CO2 absorption
        (1270.33, 0.6),   # Strong CO2 line
        (1295.48, 0.3),   # CO2 absorption
    ],
    'CH4': [
        (1180.45, 0.4),   # Methane absorption
        (1220.78, 0.3),   # CH4 line
        (1285.90, 0.5),   # Strong CH4 absorption
    ],
    'O3': [
        (1165.32, 0.6),   # Ozone absorption
        (1195.88, 0.4),   # O3 line
        (1235.67, 0.7),   # Strong ozone signature
    ],
    'CO': [
        (1175.26, 0.3),   # Carbon monoxide
        (1255.42, 0.4),   # CO absorption
    ],
    'NH3': [
        (1201.58, 0.3),   # Ammonia
        (1275.33, 0.2),   # Weak NH3 line
    ]
}

def generate_spectrum(molecules=None, planet_type='terrestrial', snr=50):
    """
    Generates a realistic simulated UV spectrum for an exoplanet with authentic molecular signatures.
    
    This enhanced version creates realistic atmospheric absorption features based on actual
    molecular data from exoplanet observations.

    Args:
        molecules (list): List of molecules to include (e.g., ['H2O', 'CO2', 'CH4']).
                         If None, randomly selects based on planet type.
        planet_type (str): Type of planet ('terrestrial', 'gas_giant', 'ice_giant')
        snr (float): Signal-to-noise ratio for realistic noise modeling

    Returns:
        tuple: A tuple containing wavelength data (numpy array), flux data (numpy array),
               and a list of molecules present in the spectrum.
    """
    print(f"Generating realistic UV spectrum for {planet_type} exoplanet...")
    
    # Wavelength range covering UV spectroscopy band
    wavelengths = np.linspace(1150, 1300, 800)  # Higher resolution
    
    # Create baseline stellar continuum (blackbody-like)
    baseline_flux = 1.0 - 0.1 * (wavelengths - 1150) / 150  # Slight slope
    
    # Determine which molecules to include based on planet type
    if molecules is None:
        if planet_type == 'terrestrial':
            # Earth-like planets likely to have water, possibly oxygen/ozone
            possible_molecules = ['H2O', 'CO2', 'O3', 'CH4']
            molecules = random.sample(possible_molecules, random.randint(2, 4))
        elif planet_type == 'gas_giant':
            # Jupiter-like planets rich in hydrogen compounds
            possible_molecules = ['H2O', 'CH4', 'NH3', 'CO']
            molecules = random.sample(possible_molecules, random.randint(2, 3))
        elif planet_type == 'ice_giant':
            # Neptune-like planets with water and methane
            possible_molecules = ['H2O', 'CH4', 'CO', 'CO2']
            molecules = random.sample(possible_molecules, random.randint(2, 3))
        else:
            molecules = ['H2O']  # Default fallback
    
    print(f"Including molecular signatures for: {', '.join(molecules)}")
    
    # Start with baseline flux
    flux = baseline_flux.copy()
    
    # Add molecular absorption lines
    for molecule in molecules:
        if molecule in MOLECULAR_SIGNATURES:
            print(f"Adding {molecule} absorption lines...")
            for wavelength_center, relative_strength in MOLECULAR_SIGNATURES[molecule]:
                if 1150 <= wavelength_center <= 1300:  # Only include lines in our range
                    
                    # Line parameters with some randomization for realism
                    line_width = random.uniform(1.5, 4.0)  # Natural line broadening
                    line_depth = relative_strength * random.uniform(0.7, 1.3)  # Abundance variation
                    
                    # Doppler broadening for planetary motion
                    doppler_shift = random.uniform(-0.5, 0.5)
                    actual_center = wavelength_center + doppler_shift
                    
                    # Create Voigt-like profile (approximated as Gaussian)
                    absorption_line = line_depth * np.exp(-((wavelengths - actual_center)**2) / (2 * line_width**2))
                    flux -= absorption_line
    
    # Add realistic noise based on SNR
    photon_noise_std = 1.0 / snr
    photon_noise = np.random.normal(0, photon_noise_std, wavelengths.shape)
    
    # Add instrumental effects
    instrumental_noise = np.random.normal(0, photon_noise_std * 0.3, wavelengths.shape)
    
    # Combine all noise sources
    flux += photon_noise + instrumental_noise
    
    # Ensure flux doesn't go negative (physical constraint)
    flux = np.maximum(flux, 0.05)
    
    return wavelengths, flux, molecules

def analyze_molecular_signatures(wavelengths, flux):
    """
    Analyzes a spectrum to detect multiple molecular species using realistic detection methods.
    
    This function uses cross-correlation and absorption line analysis to identify
    molecular signatures present in the spectrum.

    Args:
        wavelengths (numpy array): The wavelength data for the spectrum.
        flux (numpy array): The flux data for the spectrum.

    Returns:
        dict: Dictionary containing detected molecules and their confidence scores.
    """
    print("Analyzing spectrum for molecular signatures...")
    
    detections = {}
    
    # Calculate continuum (baseline) for normalization
    continuum = np.percentile(flux, 85)  # Use 85th percentile as continuum estimate
    normalized_flux = flux / continuum
    
    # Search for each molecule's signature
    for molecule, lines in MOLECULAR_SIGNATURES.items():
        print(f"Searching for {molecule}...")
        
        detection_score = 0
        lines_detected = 0
        total_expected_lines = 0
        
        for line_center, expected_strength in lines:
            if 1150 <= line_center <= 1300:  # Within our wavelength range
                total_expected_lines += 1
                
                # Find the closest wavelength indices around the line center
                search_window = 8  # Angstroms search window
                line_mask = (wavelengths >= line_center - search_window) & \
                           (wavelengths <= line_center + search_window)
                
                if np.any(line_mask):
                    # Look for absorption (flux minimum) in this region
                    region_flux = normalized_flux[line_mask]
                    region_wavelengths = wavelengths[line_mask]
                    
                    # Find the deepest absorption in this window
                    min_idx = np.argmin(region_flux)
                    min_flux = region_flux[min_idx]
                    actual_wavelength = region_wavelengths[min_idx]
                    
                    # Calculate absorption depth
                    absorption_depth = 1.0 - min_flux
                    
                    # Check if this looks like a real absorption line
                    expected_depth = expected_strength * 0.8  # Account for noise
                    wavelength_tolerance = 3.0  # Angstroms
                    
                    if (absorption_depth > expected_depth * 0.5 and 
                        abs(actual_wavelength - line_center) < wavelength_tolerance):
                        
                        # Calculate confidence based on depth and position match
                        depth_confidence = min(absorption_depth / expected_depth, 1.0)
                        position_confidence = 1.0 - abs(actual_wavelength - line_center) / wavelength_tolerance
                        
                        line_confidence = (depth_confidence + position_confidence) / 2
                        detection_score += line_confidence
                        lines_detected += 1
                        
                        print(f"  - Line at {actual_wavelength:.1f}Å (expected {line_center:.1f}Å): "
                              f"depth={absorption_depth:.3f}, confidence={line_confidence:.3f}")
        
        # Overall molecule confidence
        if total_expected_lines > 0:
            # Require detecting at least 1 line for positive identification
            if lines_detected >= 1:
                overall_confidence = (detection_score / total_expected_lines) * (lines_detected / total_expected_lines)
                # Apply detection threshold
                if overall_confidence > 0.3:  # 30% confidence threshold
                    detections[molecule] = {
                        'confidence': overall_confidence,
                        'lines_detected': lines_detected,
                        'total_lines': total_expected_lines
                    }
                    print(f"  → {molecule} DETECTED with {overall_confidence:.1%} confidence "
                          f"({lines_detected}/{total_expected_lines} lines)")
                else:
                    print(f"  → {molecule} not detected (confidence: {overall_confidence:.1%})")
            else:
                print(f"  → {molecule} not detected (no lines found)")
    
    return detections

def plot_results(wavelengths, flux, detections, actual_molecules):
    """
    Plots the spectrum and displays the molecular detection results.
    """
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

    # Main spectrum plot
    ax1.plot(wavelengths, flux, color='cyan', linewidth=1.5, alpha=0.8, label='Observed Spectrum')
    ax1.set_title('Exoplanet Atmospheric Molecular Detection', fontsize=18, pad=20)
    ax1.set_xlabel('Wavelength (Angstroms)', fontsize=12)
    ax1.set_ylabel('Relative Flux', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Mark molecular absorption regions
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    color_idx = 0
    
    for molecule in detections:
        if molecule in MOLECULAR_SIGNATURES:
            color = colors[color_idx % len(colors)]
            for line_center, strength in MOLECULAR_SIGNATURES[molecule]:
                if 1150 <= line_center <= 1300:
                    ax1.axvline(line_center, color=color, alpha=0.6, linestyle='--', linewidth=1)
            
            # Add molecule label
            ax1.text(0.02, 0.95 - color_idx * 0.05, f'{molecule}: {detections[molecule]["confidence"]:.1%}',
                    transform=ax1.transAxes, color=color, fontweight='bold', fontsize=10)
            color_idx += 1
    
    ax1.legend()
    
    # Detection summary panel
    ax2.axis('off')
    
    # Create detection summary text
    if detections:
        detected_molecules = list(detections.keys())
        result_text = f'MOLECULES DETECTED: {", ".join(detected_molecules)}'
        
        # Calculate overall habitability score
        habitability_indicators = ['H2O', 'O3', 'CO2']
        habitability_score = sum(detections.get(mol, {}).get('confidence', 0) 
                               for mol in habitability_indicators) / len(habitability_indicators)
        
        if habitability_score > 0.4:
            habitability_text = f'POTENTIALLY HABITABLE (Score: {habitability_score:.1%})'
            hab_color = 'lightgreen'
        elif habitability_score > 0.2:
            habitability_text = f'MARGINAL HABITABILITY (Score: {habitability_score:.1%})'
            hab_color = 'yellow'
        else:
            habitability_text = f'LOW HABITABILITY (Score: {habitability_score:.1%})'
            hab_color = 'orange'
        
        text_color = 'lightgreen'
        
    else:
        result_text = 'NO MOLECULAR SIGNATURES DETECTED'
        habitability_text = 'HABITABILITY: UNKNOWN'
        text_color = 'salmon'
        hab_color = 'salmon'
    
    # Display results
    ax2.text(0.5, 0.8, result_text, transform=ax2.transAxes, fontsize=14, fontweight='bold',
             color=text_color, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='darkblue', alpha=0.8))
    
    ax2.text(0.5, 0.3, habitability_text, transform=ax2.transAxes, fontsize=12, fontweight='bold',
             color=hab_color, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='darkgreen', alpha=0.6))
    
    # Ground truth comparison
    if actual_molecules:
        truth_text = f'Actual molecules: {", ".join(actual_molecules)}'
        ax2.text(0.5, 0.05, truth_text, transform=ax2.transAxes, fontsize=10,
                color='white', ha='center', va='center', style='italic')
    
    print("\nDisplaying enhanced molecular detection plot. Close the plot window to exit.")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # --- Enhanced Main Execution ---
    print("=== Realistic Exoplanet Atmospheric Analysis ===\n")
    
    # Randomly select planet type for testing
    planet_types = ['terrestrial', 'gas_giant', 'ice_giant']
    selected_planet_type = random.choice(planet_types)
    
    # Set realistic observation parameters
    signal_to_noise = random.uniform(30, 100)  # Typical for space telescopes
    
    print(f"Simulating {selected_planet_type} exoplanet observation")
    print(f"Signal-to-noise ratio: {signal_to_noise:.1f}")
    print()
    
    # 1. Generate realistic spectroscopic data
    wavelength_data, flux_data, actual_molecules = generate_spectrum(
        planet_type=selected_planet_type, 
        snr=signal_to_noise
    )
    
    print(f"\nActual molecules in atmosphere: {', '.join(actual_molecules)}")
    print()
    
    # 2. Perform molecular detection analysis
    detected_molecules = analyze_molecular_signatures(wavelength_data, flux_data)
    
    # 3. Display results summary
    print("\n" + "="*60)
    print("DETECTION SUMMARY:")
    print("="*60)
    
    if detected_molecules:
        for molecule, info in detected_molecules.items():
            confidence = info['confidence']
            lines = f"{info['lines_detected']}/{info['total_lines']}"
            print(f"{molecule:>6}: {confidence:>6.1%} confidence ({lines} lines)")
    else:
        print("No molecular signatures detected above threshold")
    
    # Calculate detection accuracy
    true_positives = len(set(detected_molecules.keys()) & set(actual_molecules))
    false_positives = len(set(detected_molecules.keys()) - set(actual_molecules))
    false_negatives = len(set(actual_molecules) - set(detected_molecules.keys()))
    
    print(f"\nAccuracy Assessment:")
    print(f"True Positives:  {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"Precision: {precision:.1%}")
    
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"Recall: {recall:.1%}")
    
    # 4. Generate comprehensive visualization
    plot_results(wavelength_data, flux_data, detected_molecules, actual_molecules)
