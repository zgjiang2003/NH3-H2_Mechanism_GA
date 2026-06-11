# NH3-H2 Mechanism Reduction & Optimization via Genetic Algorithm

Reproduction of: **Liu et al. (2023) "Development of reduced and optimized mechanism for ammonia/hydrogen mixture based on genetic algorithm"**, Fuel, 349, 128634.

## Overview

This project reproduces the complete algorithmic pipeline from the paper:

1. **Sensitivity analysis** of the detailed NH3/H2 mechanism (Stagni 2020, 34 species, 256 reactions)
2. **GA-based mechanism reduction** — binary-encoded genetic algorithm to select a minimal reaction subset
3. **GA-based mechanism optimization** — float-encoded genetic algorithm to tune Arrhenius pre-exponential factors
4. **Comprehensive validation** — ignition delay time, laminar flame speed, JSR species concentrations, premixed flame structure

## Project Structure

```
├── mechanisms/                  # Chemical mechanism files (Cantera YAML)
│   ├── Full mechanism.yaml      # Detailed mechanism (34 species, 256 reactions)
│   ├── Reduced mechanism.yaml   # GA-reduced mechanism (29 species, 63 reactions)
│   └── Optimized mechanism.yaml # GA-optimized mechanism (29 species, 63 reactions)
├── scripts/                     # Simulation & plotting scripts
│   ├── 01_idt.py                # Ignition delay time simulation (Fig. 9, 10, S1, S2)
│   ├── 02_lfs.py                # Laminar flame speed simulation (Fig. 11-15, S3-S5)
│   ├── 03_jsr.py                # JSR species concentration simulation (Fig. 16)
│   ├── 04_premixed_flame.py     # Premixed flame structure simulation (Fig. 17, 18, S6)
│   ├── 05_sensitivity.py        # Sensitivity analysis (Fig. 6)
│   ├── 06_ga_reduction.py       # GA mechanism reduction (Section 2.2)
│   ├── 07_ga_optimization.py    # GA mechanism optimization (Section 2.3)
│   ├── experimental_data.py     # Approximate experimental data from paper figures
│   ├── draw.py                  # IDT plotting (Fig. 9, 10)
│   ├── draw_lfs.py              # LFS plotting (Fig. 11-15)
│   ├── draw_jsr.py              # JSR plotting (Fig. 16)
│   ├── draw_flame.py            # Flame structure plotting (Fig. 17, 18)
│   ├── draw_sensitivity.py      # Sensitivity plotting (Fig. 6)
│   └── draw_appendix.py         # Appendix figure plotting (Fig. S1-S6)
├── results/                     # Output data & figures (generated)
│   ├── *.csv                    # Simulation data
│   └── *.png                    # Figures
└── word/                        # Reference paper & appendix
```

## Requirements

- Python 3.9+
- Cantera 3.1+
- NumPy, Pandas, Matplotlib, PyYAML

```bash
pip install cantera numpy pandas matplotlib pyyaml
```

## Usage

### 1. Run Simulations

```bash
# Ignition delay time (Fig. 9, 10, S1, S2)
python scripts/01_idt.py

# Laminar flame speed (Fig. 11-15, S3-S5)
python scripts/02_lfs.py

# JSR species concentrations (Fig. 16)
python scripts/03_jsr.py

# Premixed flame structure (Fig. 17, 18, S6)
python scripts/04_premixed_flame.py

# Sensitivity analysis (Fig. 6)
python scripts/05_sensitivity.py
```

### 2. Run GA Algorithms

```bash
# GA mechanism reduction (binary encoding, 60 population, 1000 generations)
python scripts/06_ga_reduction.py

# GA mechanism optimization (float encoding, 30 population, 1000 generations)
python scripts/07_ga_optimization.py
```

> **Note**: GA algorithms require extensive Cantera simulations per individual per generation. Runtime can range from hours to days depending on hardware.

### 3. Generate Figures

```bash
python scripts/draw.py              # Fig. 9, 10
python scripts/draw_lfs.py          # Fig. 11-15
python scripts/draw_jsr.py          # Fig. 16
python scripts/draw_flame.py        # Fig. 17, 18
python scripts/draw_sensitivity.py  # Fig. 6
python scripts/draw_appendix.py     # Fig. S1-S6
```

## Algorithm Details

### GA Mechanism Reduction (Section 2.2)

| Parameter | Value |
|-----------|-------|
| Encoding | Binary (1=keep, 0=remove) |
| Population | 60 |
| Generations | 1000 |
| Crossover | Single-point, rate=0.2 |
| Mutation | Bidirectional, rate=0.03 |
| Selection | Tournament + Elitism |

**Objective function**: f = f_tau + f_Ts + f_N + f_CPU

- f_tau: IDT error (logarithmic, sigma=1)
- f_Ts: Steady-state temperature error (logarithmic, sigma=1)
- f_N: Mechanism size penalty (sigmoid, sigma=6, lambda=0.8)
- f_CPU: Computational cost penalty (sigmoid, sigma=6, lambda=4)

**Target conditions**: Table 3 Case 1-8 (XH2=0/0.05/0.3/0.7, P=1/10 atm)

### GA Mechanism Optimization (Section 2.3)

| Parameter | Value |
|-----------|-------|
| Encoding | Float (A-factor multiplier) |
| Population | 30 |
| Generations | 1000 |
| Crossover | Single-point, rate=0.3 |
| Mutation | Bidirectional, rate=0.06 |
| Selection | Tournament + Elitism |
| A-factor range | [0.1*A_i, 10*A_i] |

**Objective function**: f = f_tau + f_LFS + f_A + f_CPU

- f_tau: IDT error (sigma=4)
- f_LFS: Laminar flame speed error (linear, sigma=4)
- f_A: A-factor deviation penalty (sigma=0.3)
- f_CPU: Computational cost penalty (sigma=6, lambda=4)

**Excluded from optimization**: H2 sub-mechanism reactions and pressure-dependent reactions.

## Simulation Conditions (Table 3)

### Reduction Targets (Case 1-8)

| Case | X[H2] | P (atm) | T (K) | phi |
|------|-------|---------|-------|-----|
| 1 | 0 | 1 | 1540-1850 | 1 |
| 2 | 0 | 10 | 1430-1820 | 1 |
| 3 | 0.05 | 1 | 1500-1920 | 1 |
| 4 | 0.05 | 10 | 1450-1920 | 1 |
| 5 | 0.3 | 1 | 1180-1820 | 1 |
| 6 | 0.3 | 10 | 1180-1920 | 1 |
| 7 | 0.7 | 1 | 1000-1540 | 1 |
| 8 | 0.7 | 10 | 1030-1300 | 1 |

### Optimization Targets (Case 9-12)

| Case | X[H2] | P (atm) | T (K) | phi |
|------|-------|---------|-------|-----|
| 9 | 0 | 1 | 423 | 0.7-1.4 |
| 10 | 0.4 | 1-5 | 298 | 0.7-1.5 |
| 11 | 0-1.0 | 1 | 298 | 1 |
| 12 | 0.05 | 1 | 298-473 | 0.8-1.4 |

## Figures Reproduced

| Paper Figure | Script | Description |
|-------------|--------|-------------|
| Fig. 6 | `05_sensitivity.py` + `draw_sensitivity.py` | Normalized IDT sensitivity (8 conditions) |
| Fig. 9 | `01_idt.py` + `draw.py` | IDT vs 1000/T, XH2=0/0.05/0.3/0.7 |
| Fig. 10 | `01_idt.py` + `draw.py` | IDT, NH3/air + NH3/O2/Ar |
| Fig. 11-12 | `02_lfs.py` + `draw_lfs.py` | LFS vs phi, various XH2 |
| Fig. 13 | `02_lfs.py` + `draw_lfs.py` | LFS vs XH2, P=1/3/5 atm |
| Fig. 14 | `02_lfs.py` + `draw_lfs.py` | LFS vs phi, XH2=0.4, P=1/3/5 atm |
| Fig. 15 | `02_lfs.py` + `draw_lfs.py` | LFS vs XH2, various Tu |
| Fig. 16 | `03_jsr.py` + `draw_jsr.py` | JSR species (NH3, H2O, NO, N2O) |
| Fig. 17 | `04_premixed_flame.py` + `draw_flame.py` | Premixed flame structure |
| Fig. 18 | `04_premixed_flame.py` + `draw_flame.py` | NH3/NO/Ar flame structure |
| Fig. S1-S6 | `01-04_*.py` + `draw_appendix.py` | Appendix validation figures |

## Known Limitations

1. **Fig. 18 (experimental temperature input)**: The paper uses experimental temperature profiles as simulation input. Cantera does not natively support fixed temperature fields in 1D flames; FreeFlame is used as an approximation.
2. **Fig. 19-20 (HCCI engine)**: Requires 3D CFD software (CONVERGE), beyond Cantera's scope.
3. **Fig. S3 (P=40 atm LFS)**: FreeFlame convergence at 40 atm is extremely difficult; this condition is skipped.
4. **Experimental data**: Values in `experimental_data.py` are approximated from paper figures, not precisely digitized.

## Reference

Liu, Y., Zhang, X., & Qi, F. (2023). Development of reduced and optimized mechanism for ammonia/hydrogen mixture based on genetic algorithm. *Fuel*, 349, 128634. https://doi.org/10.1016/j.fuel.2023.128634

## License

This project is for academic research purposes only. The original mechanism is from Stagni et al. (2020) and the optimization methodology follows Liu et al. (2023).
