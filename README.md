# Social Contagion Simulator

Local Streamlit app for a Complex Systems presentation on social contagion in
networks. It compares simple SIR/SIS/SEIR contagion with complex threshold
contagion across undirected networks and directed influencer networks.

## Features

- Synthetic networks with NetworkX:
  - Erdos-Renyi
  - Watts-Strogatz
  - Barabasi-Albert
  - Scale-Free
  - Directed Influencers
- Directed influencer networks where arrows represent one-way influence.
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
- Mathematics tab with the transition equations, current parameter values, and
  derived pressure metrics for the active run.
- Initial, peak, and final network snapshots.
- S(t), E(t), I(t), R(t) curves.
- Final cascade size, peak infected fraction, and time to peak.
- Automatic beta and theta sweeps.
- PNG export from the app.

## Download the Repository

With Git:

```bash
git clone https://github.com/jenriquezafra/social-contagion.git
cd social-contagion
```

Without Git, download the ZIP from:

```text
https://github.com/jenriquezafra/social-contagion
```

Then unzip it and open a terminal inside the extracted `social-contagion`
folder.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

If port `8501` is busy:

```bash
streamlit run app.py --server.port 8502
```

If you do not activate the virtual environment, run Streamlit directly from it:

```bash
.venv/bin/streamlit run app.py
```

## Parameters

### Appearance

| Parameter | Values | Default | Meaning |
| --- | --- | --- | --- |
| `Visual theme` | `Auto`, `Light`, `Dark` | `Auto` | Matches the page and simulation stage palette. |
| `Stage format` | `Wide`, `Compact`, `Tall` | `Wide` | Sets the presentation-oriented canvas shape. |
| `Stage height` | `460` to `820` | preset-dependent | Pixel height of the interactive stage. |
| `Stage width` | `70` to `100` | `100` | Percentage width of the stage in the page. |

### Network

| Parameter | Values | Default | Meaning |
| --- | --- | --- | --- |
| `Number of nodes` | `20` to `500` | `150` | Number of agents in the graph. |
| `Average degree` | `1` to `30` | `6` | Target average number of neighbors per node. |
| `Topology` | `Erdos-Renyi`, `Watts-Strogatz`, `Barabasi-Albert`, `Scale-Free`, `Directed Influencers` | `Erdos-Renyi` | Network generation model. |
| `Influencer fraction` | `0.01` to `0.25` | `0.06` | Only shown for `Directed Influencers`; fraction of nodes with outgoing influence and zero incoming social ties. |

### Initial Condition

| Parameter | Values | Default | Meaning |
| --- | --- | --- | --- |
| `Initial infected` | `1` to `min(80, n_nodes)` | `5` | Number of infected seed nodes at `t=0`. |
| `Seed mode` | `random`, `hubs` | `random` | Choose seeds randomly or from strongest influence hubs; directed graphs rank by out-degree. |
| `Maximum steps` | `5` to `200` | `60` | Number of synchronous simulation updates. |
| `Random seed` | integer `>= 0` | `42` | Reproduces the network and stochastic draws. |

### Model

| Parameter | Values | Default | Meaning |
| --- | --- | --- | --- |
| `Contagion model` | `simple`, `threshold` | `simple` | Selects probabilistic simple contagion or complex threshold contagion. |
| `Simple model type` | `SIR`, `SIS`, `SEIR` | `SIR` | State-transition variant used when `model=simple`. |
| `beta` | `0.00` to `1.00` | `0.30` | Per-neighbor transmission probability in simple contagion. |
| `sigma` | `0.00` to `1.00` | `0.35` | Probability that an exposed SEIR node becomes infected per step. |
| `theta` | `0.00` to `1.00` | `0.25` | Infected-neighbor fraction needed for threshold adoption. |
| `gamma` | `0.00` to `1.00` | `0.10` | Probability that an infected node recovers per step. |
| `external_noise` | `0.000` to `0.100` | `0.000` | Outside adoption probability per susceptible node per step. |

The app also runs two automatic sweeps for comparison:

```text
beta sweep:  0.1, 0.2, 0.3, 0.4, 0.5
theta sweep: 0.1, 0.25, 0.4
```

## Project Structure

```text
social-contagion/
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

The app uses synchronous updates: every node reads the state at time `t`, then
all node states are written to `t+1` together.

For node `i`:

```text
m_i(t) = number of infected neighbors
k_i    = total number of neighbors
epsilon = external_noise
```

In undirected graphs, neighbors are ordinary two-way ties. In `Directed
Influencers`, an edge `u -> v` means `u` can influence `v`; therefore `m_i(t)`
counts infected predecessors and `k_i` is the in-degree of node `i`.
Influencer nodes have no incoming social ties, so they are not infected by
network contagion. They can still be initial seeds or adopt through
`external_noise`.

Simple contagion uses the per-step neighbor probability:

```text
q_i(t) = 1 - (1 - beta)^m_i(t)
P(contagion) = q_i(t) + (1 - q_i(t)) * epsilon
```

Modes:

- SIR: infected nodes recover to `R`.
- SIS: infected nodes recover to `S`.
- SEIR: new contagions enter `E`, exposed nodes become infected with probability
  `sigma`, and infected nodes recover to `R`.

Threshold contagion infects a susceptible node when:

```text
m_i(t) / k_i >= theta
```

If the threshold condition is not met, external noise can still trigger adoption
with probability `epsilon`.

Both models include recovery with probability `gamma` per infected node per
step.

`Final cascade size` is measured as the cumulative fraction of nodes that were
infected or exposed at least once, which keeps the metric meaningful for SIR,
SIS, SEIR, and threshold cascades.

The in-app `Mathematics` tab shows these equations, the current parameter
values, a rough simple-contagion early-spread heuristic, and peak-step pressure
metrics computed from the active network.
