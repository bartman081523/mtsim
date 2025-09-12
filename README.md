# MTSim - Microtubule Simulation Framework

A comprehensive simulation framework for studying microtubule dynamics and associated photonic emission phenomena. This project combines classical microtubule mechanics with quantum optical modeling to investigate coherent vs incoherent emission patterns in biological systems.

## Overview

MTSim provides a hybrid simulation environment that models:
- **Photonic Emission**: Superradiant (coherent) vs incoherent emission from microtubule-associated emitters
- **Microtubule Dynamics**: Four-state model (grow/shrink/pause/transition) with realistic kinetic parameters
- **Detector Physics**: Quantum efficiency, geometric losses, dark counts, and time-gated detection
- **Network Topologies**: Multi-emitter networks with varying connectivity and coupling

## Key Features

### Photonic Modeling
- **Hypothesis Mode**: Superradiant emission with enhanced temporal coherence (Dicke-like sech² pulses)
- **Antithesis Mode**: Incoherent emission with exponential decay kinetics
- **Curvature Effects**: Geometric enhancement/suppression of emission coupling
- **Peak Gating**: Sub-nanosecond temporal windows for enhanced discrimination

### Detection System
- Realistic detector modeling with quantum efficiency, dark counts, and afterpulsing
- Geometric collection efficiency with distance-dependent attenuation
- Signal-to-noise ratio optimization for weak signals
- Time-resolved detection with configurable gate widths

### Parameter Sweeps
- Automated exploration of parameter space for optimization
- Network topology effects on collective emission
- Distance-dependent coupling studies
- Detector performance characterization

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/bartman081523/mtsim.git
cd mtsim

# Set up Python environment
python -m venv .venv-mtsim
source .venv-mtsim/bin/activate  # On Windows: .venv-mtsim\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Full Setup with External Dependencies

```bash
# Run the complete setup (includes external repos)
chmod +x setup.sh
./setup.sh
```

This will create a `mt_hybrid` directory with external repositories for:
- Spindle simulation (Lera-Ramirez 2021)
- kMC motor protein dynamics
- VCell integration
- Four-state MT models
- Quantum microtubule simulations

## Usage

### Basic Simulation

Run a single hybrid simulation comparing hypothesis vs antithesis modes:

```bash
python run_hybrid_sim.py --curvature 1.0 --loss-mm 10.0 --r-um 50.0 --trials 8
```

**Parameters:**
- `--curvature`: Geometric coupling enhancement factor (default: 1.0)
- `--loss-mm`: Attenuation length in mm (default: 10.0)
- `--r-um`: Detection distance in micrometers (default: 50.0)
- `--trials`: Number of simulation trials for averaging (default: 8)
- `--seed`: Random seed for reproducibility (default: 1234)
- `--win-ns`: Gate window width in nanoseconds (default: 0.5)
- `--deadtime-ns`: Detector deadtime in nanoseconds (default: 0.0)
- `--afterpulse`: Afterpulse probability (default: 0.0)
- `--qe`: Quantum efficiency (default: 0.7)
- `--eta-geom`: Geometric collection efficiency (default: 0.2)
- `--dark-cps`: Dark count rate in counts/second (default: 50.0)
- `--emission-scale`: Emission scaling factor (default: 1.0)

### Parameter Sweeps

Explore parameter space systematically:

```bash
python run_param_sweep.py --preset sweetspot --out results.csv
```

**Parameters:**
- `--preset`: Choose parameter preset ("default" or "sweetspot")
- `--curvatures`: Comma-separated curvature values (e.g., "0.5,1.0,1.5")
- `--losses`: Comma-separated loss values in mm⁻¹ (e.g., "5,10,30")
- `--distances`: Comma-separated distances in μm (e.g., "10,50,100")
- `--trials`: Number of simulation trials (default: 16)
- `--seed`: Random seed for reproducibility (default: 1000)
- `--window-ns`: Gate window width in nanoseconds (default: 0.5)
- `--optimize-window`: Comma-separated window values for optimization
- `--deadtime-ns`: Detector deadtime in nanoseconds (default: 0.0)
- `--afterpulse`: Afterpulse probability (default: 0.0)
- `--qe`: Quantum efficiency (default: 0.6)
- `--eta-geom`: Geometric collection efficiency (default: 0.1)
- `--dark-cps`: Dark count rate in counts/second (default: 100.0)
- `--emission-scale`: Emission scaling factor (default: 1.0)
- `--out`: Output CSV filename (default: "param_sweep_results.csv")
- `--no-header`: Skip CSV header when appending to existing file

### Network Simulations

Study collective effects in emitter networks:

```bash
python run_network_sweep.py --preset sweetspot --out network_results.csv
```

**Parameters:**
- `--preset`: Choose parameter preset ("default" or "sweetspot")
- `--N`: Number of network nodes (default: 12)
- `--node-radius-um`: Node radius in micrometers (default: 12.0)
- `--det-pos-um`: Detector position as "x,y,z" in μm (default: "0,0,20")
- `--curvatures`: Comma-separated curvature values (default: "0.5,1.0")
- `--losses`: Comma-separated loss values in mm⁻¹ (default: "5,10,30")
- `--alphas`: Comma-separated alpha coupling values (default: "0.1,0.3")
- `--ells-um`: Comma-separated correlation lengths in μm (default: "8,12")
- `--trials`: Number of simulation trials (default: 12)
- `--seed`: Random seed for reproducibility (default: 1000)
- `--window-ns`: Gate window width in nanoseconds (default: 0.5)
- `--optimize-window`: Comma-separated window values for optimization
- `--deadtime-ns`: Detector deadtime in nanoseconds (default: 0.0)
- `--afterpulse`: Afterpulse probability (default: 0.0)
- `--qe`: Quantum efficiency (default: 0.6)
- `--eta-geom`: Geometric collection efficiency (default: 0.1)
- `--dark-cps`: Dark count rate in counts/second (default: 100.0)
- `--kappa-sigma`: Curvature deviation parameter (default: 0.0)
- `--emission-scale`: Emission scaling factor (default: 1.0)
- `--prc-eps`: PRC coupling strength (default: 0.2)
- `--out`: Output CSV filename (default: "sweep_network.csv")
- `--no-header`: Skip CSV header when appending to existing file

## File Structure

```
mtsim/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.sh                     # Complete setup script
│
├── mt_upe_core.py              # Core simulation engine
├── run_hybrid_sim.py           # Single simulation runner
├── run_param_sweep.py          # Parameter sweep automation
├── run_network_sweep.py        # Network simulation runner
│
├── modules/                    # Core simulation modules
│   ├── photonic.py            # Photonic emission models
│   ├── detector.py            # Detector physics simulation
│   └── spin_ros.py            # Spin-related calculations
│
├── adapters/                  # External system adapters
│   ├── mt_spindle.py         # Four-state microtubule model
│   └── kmc_motors.py         # Motor protein kinetics
│
├── sweep_*.csv               # Parameter sweep configurations
├── param_sweep_results.csv   # Example results
└── external/                 # External repositories (after setup.sh)
```

## Physical Parameters

### Photonic Constants
- **Wavelength**: 280 nm (UV)
- **Photon Energy**: ~4.4 eV
- **Spontaneous Lifetime**: 1 ns
- **Default Time Resolution**: 1 ps
- **Simulation Duration**: 5 ns

### Microtubule Dynamics
- **Growth Velocity**: 200-600 nm/s
- **Shrinkage Velocity**: 500-1000 nm/s
- **Catastrophe Rate**: ~0.2 s⁻¹
- **Rescue Rate**: ~0.05 s⁻¹

### Detection Parameters
- **Quantum Efficiency**: 0.6-0.8
- **Geometric Collection**: 0.1-0.3
- **Dark Count Rate**: 20-100 cps
- **Gate Width**: 0.1-1.0 ns

## Scientific Background

This simulation framework is designed to investigate the following hypotheses:

1. **Coherent Emission Hypothesis**: Microtubule networks can support superradiant emission through quantum coherence, leading to temporally sharp, high-intensity pulses with enhanced signal-to-noise ratios.

2. **Incoherent Emission Antithesis**: Biological systems exhibit only classical incoherent emission with exponential decay kinetics and broad temporal distributions.

The framework provides quantitative metrics to discriminate between these scenarios:
- **Peak Sharpness**: Ratio of peak to mean intensity
- **Peak-to-RMS**: Enhanced contrast metric
- **Signal-to-Noise**: Detection fidelity under realistic noise conditions

## Results Analysis

The simulation generates CSV files with the following key metrics:
- `mode`: "hypothesis" (superradiant) or "antithesis" (incoherent)
- `peak_sharp`: Peak sharpness metric
- `peak_rms`: Peak-to-RMS ratio
- `snr`: Signal-to-noise ratio
- `N_detected`: Detected photon count
- `discrimination`: Statistical separability between modes

## Contributing

This is a research simulation framework. When contributing:
1. Maintain physical realism in all models
2. Document parameter choices with literature references
3. Validate new features against analytical limits
4. Include uncertainty quantification in results

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Scientific algorithms
- `networkx`: Network topology analysis
- `matplotlib`: Visualization
- `pandas`: Data analysis
- `tqdm`: Progress bars

## License

This project is released under an open research license. Please cite appropriately if used in scientific publications.

## Contact

For questions about the simulation framework or collaboration opportunities, please open an issue in the repository.