import numpy as np

currentTurn = -1
currentBoard = 4
RUNS = 0
MOVES = 0

points = np.array([0.2, 0.17, 0.2, 0.17, 0.22, 0.17, 0.2, 0.17, 0.2])
evaluatorMul = np.array([1.4, 1, 1.4, 1, 1.75, 1, 1.4, 1, 1.4])

playerNames = ["PLAYER", "AI"]

line_win = [
    [0,1,2],
    [3,4,5],
    [6,7,8],
    [0,3,6],
    [1,4,7],
    [2,5,8],
    [0,4,8],
    [2,4,6],
]
