# Rock Paper Scissors (MediaPipe)

Play Rock–Paper–Scissors with your webcam. The app reads your hand gesture and the computer picks a random move.

## Setup

```powershell
pip install -r requirements.txt
python rps_game.py
```

A camera is required. On first run the MediaPipe hand model (`hand_landmarker.task`) is downloaded automatically, so you do not need to add that file to GitHub.

Click **Start Game**, wait for the countdown, then show rock, paper, or scissors.
