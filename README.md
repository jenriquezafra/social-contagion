# Social Contagion Simulator

Local Streamlit app for a Complex Systems presentation on social contagion in
networks. It compares simple SIR/SIS/SEIR contagion with complex threshold contagion.

## Features

- Synthetic networks with NetworkX:
  - Erdos-Renyi
  - Watts-Strogatz
  - Barabasi-Albert
  - Scale-Free
- Initial infected nodes selected at random or from hubs.
- Simple contagion with SIR, SIS, or SEIR dynamics.
- Complex threshold contagion.
- External noise for media, algorithms, or outside influence.
- Simultaneous updates using a separate `next_state`.
- Apple-like Streamlit page with a custom HTML/CSS/JavaScript canvas stage.
- Appearance selector with Auto, Light, and Dark modes.
- Light and dark stage palettes, so the main simulation matches the page theme.
- Stage size controls for wide, compact, or tall presentation formats.
- Step-based playback with stable node colors and a synchronized curve marker.
- Pan and zoom inside the main network canvas with drag, scroll, buttons, and double-click reset.
- Analysis tabs for snapshots, time series, parameter sweeps, and export.
- Initial, peak, and final network snapshots.
- S(t), E(t), I(t), R(t) curves.
- Final cascade size, peak infected fraction, and time to peak.
- Automatic beta and theta sweeps.
- PNG export from the app.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Project Structure

```text
social-contagion-app/
├── app.py
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── simulation.py
    ├── networks.py
    ├── plots.py
    └── web_stage.py
```

## Model Notes

Simple contagion uses:

```text
p = 1 - (1 - beta)^m
```

where `m` is the number of infected neighbors.

Modes:

- SIR: infected nodes recover to `R`.
- SIS: infected nodes recover to `S`.
- SEIR: new contagions enter `E`, exposed nodes become infected with probability
  `sigma`, and infected nodes recover to `R`.

Threshold contagion infects a susceptible node when:

```text
infected_neighbors / total_neighbors >= theta
```

Both models include recovery with probability `gamma` and optional external
noise with probability `external_noise` per susceptible node per step.

`Final cascade size` is measured as the cumulative fraction of nodes that were
infected or exposed at least once, which keeps the metric meaningful for SIR,
SIS, SEIR, and threshold cascades.
