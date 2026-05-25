# Project Setup and Sample Run

## 1) Clone this repository (Project folder)

```bash
git clone https://github.com/samrudh123/RNM-based-Private-Text-Generation.git
cd "RNM-based-Private-Text-Generation"
```

## 2) Clone required repositories inside this repo

```bash
git clone https://github.com/cerai-iitm/invisibleink.git
git clone https://github.com/samrudh123/invisibleink_x_RNM.git
git clone https://github.com/samrudh123/Weak_invisible_ink.git
```

> Replace the placeholder URLs with the actual repository links.

## 3) Set up a virtual environment and install dependencies

Windows (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Run sh files to generate the output in the respective folders

Run the script which has explicit Python calls in the repository.
