# DeepFace Emotion TCP Server

This project runs a modified DeepFace real-time emotion stream and exposes the latest emotion state over a local TCP socket for the Lyria/Gemma side to consume.

## Python version

This project was developed and tested with:

```powershell
Python 3.13.0
```

Use Python 3.13 on a new machine if possible. Python 3.12 may work, but it has not been verified for this project.

## Install Python on a new Windows machine

1. Download and install Python 3.13 from the official Python website.
2. During installation, enable **Add python.exe to PATH**.
3. Open PowerShell and confirm Python is available:

```powershell
python --version
```

Expected result:

```text
Python 3.13.x
```

## Set up the project

Clone the repository, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run PowerShell as your user and execute:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate `.venv` again.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python emotion_server_real.py
```

The server listens on:

```text
127.0.0.1:54322
```

It sends newline-delimited JSON emotion payloads to connected clients.

## Notes

- The `deepface/` package in this repository is a local modified copy of DeepFace 0.0.96.
- The modified DeepFace files are `deepface/DeepFace.py` and `deepface/modules/streaming.py`.
- Model weights and runtime cache are not stored in Git. They are ignored via `.gitignore` and may be downloaded/generated at runtime.
- Local face databases, test code, notebooks, images, and cache files are intentionally not included.
