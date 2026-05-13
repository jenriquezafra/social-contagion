# Social Contagion Simulator

Streamlit app for simulating social contagion on networks. It supports synthetic
NetworkX graphs and an optional real-data mode based on the SNAP Higgs Twitter
retweet dataset.

The simulator includes SIR, SIS, and SEIR dynamics, threshold adoption, optional
reinforcement memory, external noise, parameter sweeps, and interactive plots.

## Requirements

- Python 3.10 or newer
- `pip`

## Install

Clone the repository:

```bash
git clone https://github.com/jenriquezafra/social-contagion.git
cd social-contagion
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

If port `8501` is busy:

```bash
streamlit run app.py --server.port 8502
```

If you do not activate the virtual environment, you can run Streamlit directly:

```bash
.venv/bin/streamlit run app.py
```

## Optional Twitter Higgs Data

The Twitter Higgs data is not required to run the app. The synthetic network
modes work without any extra files.

The real Twitter mode uses two raw files from the public SNAP Higgs Twitter
dataset:

- `higgs-retweet_network.edgelist.gz`
- `higgs-activity_time.txt.gz`

They are intentionally not stored in the repository. Download them with:

```bash
python scripts/download_higgs_twitter.py
```

The files are saved under:

```text
data/higgs-twitter/raw/
```

You can also select `Twitter Higgs retweets` in the sidebar and use the
`Download Higgs data` button shown by the app when the files are missing.

If the files are not present, the app should still open normally with synthetic
topologies. Only the Twitter Higgs topology needs those files; when selected
without data, the app shows a message with the download command instead of
crashing.

SNAP dataset page:

```text
https://snap.stanford.edu/data/higgs-twitter.html
```

## Model Note

State updates are synchronous: every node reads the state at time `t`, writes
its possible transition into a separate `next_state`, and all nodes move to
`t + 1` together.

## Project Structure

```text
social-contagion/
├── app.py
├── requirements.txt
├── scripts/
│   └── download_higgs_twitter.py
└── src/
    ├── networks.py
    ├── plots.py
    ├── simulation.py
    ├── twitter_higgs.py
    └── web_stage.py
```
