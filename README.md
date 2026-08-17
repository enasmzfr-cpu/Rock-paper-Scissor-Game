# Rock Paper Scissors (MediaPipe)

Play Rock–Paper–Scissors with your webcam. The app reads your hand gesture and the computer picks a random move.
It is a small Python desktop app: rps_game.py, requirements.txt, and the MediaPipe hand_landmarker.task model.

# How to play
You click Start, wait for a 3-second countdown, then show rock, paper, or scissors. The app classifies your hand, the computer picks a random move, and the Tkinter window shows the result and score. Play Again and Reset keep the match going.



## Setup

```powershell
pip install -r requirements.txt
python rps_game.py
```

A camera is required. On first run the MediaPipe hand model (`hand_landmarker.task`) is downloaded automatically, so you do not need to add that file to GitHub.


