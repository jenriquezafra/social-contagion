# Social Contagion Simulator

Local Streamlit app for a Complex Systems presentation on social contagion in
networks. It compares simple SIR/SIS/SEIR contagion with complex threshold
adoption inside SEIR across undirected networks, with an optional directed
influencer layer on Barabasi-Albert networks, and a real Twitter retweet sample
from the SNAP Higgs rumor dataset.

## Features

- Synthetic networks with NetworkX:
  - Erdos-Renyi
  - Watts-Strogatz
  - Barabasi-Albert
  - Scale-Free
- Optional influencer layer on Barabasi-Albert networks.
- Real Twitter mode using SNAP Higgs retweets and activity from July 1-7, 2012.
- Initial infected nodes selected at random for synthetic runs or from earliest
  observed activity for Twitter runs.
- Simple contagion with SIR, SIS, or SEIR dynamics.
- Optional complex reinforcement memory for SEIR contagion.
- Threshold adoption as a SEIR exposure rule.
- External noise for media, algorithms, or outside influence.
- Simultaneous updates using a separate `next_state`.
- Apple-like Streamlit page with a custom HTML/CSS/JavaScript canvas stage.
- Appearance selector with Auto, Light, and Dark modes.
- Light and dark stage palettes, so the main simulation matches the page theme.
- Stage size controls for wide, compact, or tall presentation formats.
- Fixed experiment presets with scenario interpretation.
- Step-based playback with stable node colors and a synchronized curve marker.
- Pan and zoom inside the main network canvas with drag, scroll, buttons, and double-click reset.
- Analysis tabs for snapshots, time series, parameter sweeps, topology notes,
  mathematics, Twitter data, and export.
- Topologies tab with network-generation equations and current graph metrics.
- Mathematics tab with transition equations, current parameter values, and
  derived pressure metrics for the active run.
- Initial, peak, and final network snapshots.
- S(t), E(t), I(t), R(t) curves.
- Final cascade size, peak infected fraction, and time to peak.
- Automatic parameter sweeps for the selected model.
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

## Twitter Higgs Data

The app can use the public SNAP Higgs Twitter dataset. It models retweet
diffusion around the Higgs boson announcement from July 1-7, 2012.

Download the two small files used by the app:

```bash
python scripts/download_higgs_twitter.py
```

The sidebar also has a `Download Higgs data` button when the files are missing.
The app stores them under:

```text
data/higgs-twitter/raw/
```

For simulation, a raw retweet action `A retweets B` is reversed into the
influence edge `B -> A`, so arrows point from the source account toward the
retweeter.

Source: https://snap.stanford.edu/data/higgs-twitter.html

## Parameters

### Experiment Presets

The sidebar includes presets for presentation-ready experiments. Selecting one
loads the model and network parameters and adds an interpretation in the
`Experiments` tab. If you edit any loaded preset parameter, the app switches
back to `Custom` while keeping your edited configuration.

| Preset | Model | Interpretation |
| --- | --- | --- |
| `Breaking news broadcast (SIR)` | SIR | A public event spreads through retweets plus strong outside media/trending-topic pressure. |
| `Recurring trend churn (SIS)` | SIS | Users can leave and re-enter attention cycles, as in recurring memes or consumer trends. |
| `Latent attention (SEIR)` | SEIR probabilistic | Users can see a topic before visibly participating, delaying the observed wave. |
| `Political radicalization memory (SEIR)` | SEIR with memory | Repeated exposure accumulates before visible adoption, using a stylized influencer-heavy network. |
| `Social proof threshold (SEIR)` | SEIR threshold | Users enter exposure only after enough simultaneous social signals. |

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
| `Topology` | `Erdos-Renyi`, `Watts-Strogatz`, `Barabasi-Albert`, `Scale-Free`, `Twitter Higgs retweets` | `Erdos-Renyi` | Network structure used for exposure paths. Synthetic options generate graphs; Twitter Higgs loads a real retweet topology. |
| `Number of nodes` | `20` to `500` | `150` | Number of agents in a synthetic graph. |
| `Sample users` | `50` to `500` | `250` | Number of sampled anonymized Twitter users when `Twitter Higgs retweets` is selected. |
| `Average degree` | `1` to `30` | `6` | Target average number of neighbors per node for synthetic topologies. |
| `Enable influencer layer` | checkbox | off | Only shown for `Barabasi-Albert`; marks high-degree BA nodes as influencers and orients influence edges. |
| `Influencer fraction` | `0.01` to `0.25` | `0.06` | Only shown when the influencer layer is enabled. |
| `Peers can influence influencers` | checkbox | off | Allows peer-to-influencer edges in addition to influencer-to-peer and influencer-to-influencer edges. |
| `Minimum retweet weight` | `1` to `10` | `1` | Keeps only real retweet edges observed at least this many times. |
| `Reverse retweets into influence flow` | checkbox | on | Converts `A retweets B` into `B -> A`. |

### Initial Condition

| Parameter | Values | Default | Meaning |
| --- | --- | --- | --- |
| `Initial infected` | `1` to `min(80, n_nodes)` | `5` | Number of infected seed nodes at `t=0`. |
| `Maximum steps` | `5` to `200` | `60` | Number of synchronous simulation updates. |
| `Random seed` | integer `>= 0` | `42` | Reproduces the network and stochastic draws. |

### Model

| Parameter | Values | Default | Meaning |
| --- | --- | --- | --- |
| `Contagion model` | `SIR`, `SIS`, `SEIR` | `SIR` | State-transition model. |
| `SEIR adoption rule` | `probabilistic`, `threshold` | `probabilistic` | How susceptible nodes enter `E`; only shown for `SEIR`. |
| `beta` | `0.00` to `1.00` | `0.30` | Per-neighbor transmission probability; shown for `SIR`, `SIS`, and probabilistic `SEIR`. |
| `sigma` | `0.00` to `1.00` | `0.35` | Probability that an exposed SEIR node becomes infected per step; only shown for `SEIR`. |
| `Complex reinforcement` | checkbox | off | Probabilistic SEIR memory: susceptible nodes accumulate repeated exposures across steps. |
| `Memory retention` | `0.00` to `1.00` | `0.75` | Fraction of remembered exposure pressure kept each step; only shown when `Complex reinforcement` is enabled. |
| `theta` | `0.00` to `1.00` | `0.25` | Infected-neighbor fraction needed for threshold adoption; only shown for threshold `SEIR`. |
| `gamma` | `0.00` to `1.00` | `0.10` | Probability that an infected node recovers per step. |
| `external_noise` | `0.000` to `0.100` | `0.000` | Outside adoption probability per susceptible node per step. |

The app runs automatic sweeps only for parameters used by the selected model:

```text
SIR/SIS:   beta, gamma
SEIR probabilistic: beta, sigma, gamma
SEIR threshold:     theta, sigma, gamma
```

## Project Structure

```text
social-contagion/
├── app.py
├── scripts/
│   └── download_higgs_twitter.py
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── simulation.py
    ├── networks.py
    ├── twitter_higgs.py
    ├── plots.py
    └── web_stage.py
```

## Model Notes

The app uses synchronous updates: every node reads the state at time `t`, then
all node states are written to `t+1` together.

## Topology Notes

`n` is the number of nodes and `k_bar` is the requested average degree.

### Erdos-Renyi

```text
G(n, p)
p = k_bar / (n - 1)
E[M] = p * n * (n - 1) / 2
E[k_i] = p * (n - 1)
```

Every possible undirected edge is sampled independently. This is the neutral
baseline: no explicit high-degree mechanism, no spatial clustering, no
preferential attachment.

### Watts-Strogatz

```text
k_ring = nearest valid even degree to k_bar
p_rewire = 0.1
E[k_i] approx k_ring
```

The graph starts as a ring lattice and rewires edges with probability
`p_rewire`. It keeps local neighborhoods while adding long-range shortcuts,
which models small-world structure.

### Barabasi-Albert

```text
m = max(1, round(k_bar / 2))
P(new edge attaches to i) = k_i / sum_j(k_j)
E[k] approx 2m
P(k) ~ k^-3
```

New nodes attach preferentially to nodes that already have high degree. This
creates the heavy-tailed structure used by the optional influencer layer.

### Influencer Layer on Barabasi-Albert

```text
F = max(1, round(influencer_fraction * n))
I = top F nodes by BA degree
u -> v means u can influence v
i, j in I => i <-> j when the BA edge exists
peer -> influencer only if enabled
```

This is not a separate topology. The base graph is still Barabasi-Albert. The
layer marks the highest-degree BA nodes as influencers and then orients each BA
edge:

- influencer-peer edges always include `influencer -> peer`;
- influencer-influencer edges are bidirectional;
- peer-peer edges are bidirectional;
- peer-influencer edges are included only when `Peers can influence influencers`
  is enabled.

### Scale-Free

```text
P(k) proportional to k^-alpha
M_target approx n * k_bar / 2
P(extra source = i) = (k_i + 1) / sum_j(k_j + 1)
```

The app starts from a NetworkX scale-free graph, converts it to a simple
undirected graph, then adds preferentially weighted edges until it roughly
matches the requested average degree.

### Twitter Higgs Retweets

```text
A retweets B => B -> A
w_BA = observed retweet count
initial seeds = earliest observed Higgs activity in the sampled graph
influencers = top sampled nodes by weighted out-degree
```

This mode uses the SNAP Higgs Twitter retweet network and activity log. The
published user IDs are anonymized. The app samples a presentation-sized
subgraph around early topic users and high-outdegree retweeted accounts.

## Contagion Notes

For node `i`:

```text
m_i(t) = number of infected neighbors
k_i    = total number of neighbors
epsilon = external_noise
```

In undirected graphs, neighbors are ordinary two-way ties. In directed
influence graphs, including the Barabasi-Albert influencer layer and Twitter
Higgs retweets, an edge `u -> v` means `u` can influence `v`; therefore
`m_i(t)` counts infected predecessors and `k_i` is the in-degree of node `i`.

Simple contagion uses the per-step neighbor probability:

```text
q_i(t) = 1 - (1 - beta)^m_i(t)
P(contagion) = q_i(t) + (1 - q_i(t)) * epsilon
```

Modes:

- SIR: infected nodes recover to `R`.
- SIS: infected nodes recover to `S`.
- SEIR: new adoptions enter `E`, exposed nodes become infected with probability
  `sigma`, and infected nodes recover to `R`.

With `Complex reinforcement` enabled in SEIR, susceptible nodes remember prior
exposure pressure:

```text
rho = memory_retention
r_i(t) = min(k_i, rho * r_i(t-1) + m_i(t))
q_i(t) = 1 - (1 - beta)^r_i(t)
```

This models repeated exposure: a user who has seen the topic several times can
become exposed later even if adoption did not happen on the first contact. Low
retention forgets pressure quickly; high retention keeps past exposure pressure
alive for longer.

With threshold adoption in SEIR, a susceptible node enters `E` when:

```text
m_i(t) / k_i >= theta
```

If the threshold condition is not met, external noise can still trigger exposure
with probability `epsilon`. The path is still `S -> E -> I -> R`.

All models include recovery with probability `gamma` per infected node per
step.

`Final cascade size` is measured as the cumulative fraction of nodes that were
infected or exposed at least once, which keeps the metric meaningful for SIR,
SIS, probabilistic SEIR, and threshold-SEIR cascades.

The in-app `Topologies` tab shows topology equations and graph metrics. The
`Mathematics` tab shows contagion equations, current parameter values, a rough
simple-contagion early-spread heuristic, and peak-step pressure metrics computed
from the active network.
