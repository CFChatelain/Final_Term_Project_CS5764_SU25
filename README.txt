Airbnb NYC 2019 – README
===========================================

Project structure
-----------------
AB_NYC_2019.csv          ← raw dataset 48 k listings
phase1_static_plots.py   ← Phase I: generates 30 static figures
airbnb_nyc_project.py    ← Phase II: Dash dashboard
requirements.txt         ← exact Python packages
figures/                 ← holds PNGs
README.txt


Prerequisites
----------------
• Python 3.8 – 3.11
• pip
• virtual-env tool


Quick-start
------------------------------------
# 0) open terminal, cd into project folder
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt

# Phase I – build static plots
python phase1_static_plots.py
#  → figures/*.png  +  figures/README.md (captions)

# Phase II – launch dashboard
python airbnb_nyc_project.py          # default 127.0.0.1:8050
# or bind manually:
python airbnb_nyc_project.py --host 0.0.0.0 --port 8080
# open the printed URL in your browser


Phase III (deployment outline)
---------------------------------
Dockerfile
----------
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8050
CMD ["python", "airbnb_nyc_project.py", "--host", "0.0.0.0"]
