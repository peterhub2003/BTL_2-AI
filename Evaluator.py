import numpy as np
import math


RUNS = 0

#points = np.array([0.2, 0.17, 0.2, 0.17, 0.22, 0.17, 0.2, 0.17, 0.2])
points = np.array([0.3, 0.2, 0.3, 0.2, 0.4, 0.2, 0.3, 0.2, 0.3])

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

condition_1 = False


def checkWinCondition(map: np.array):
	for a in [1, -1]:
		for ls in line_win:
			if np.sum(map[ls]) == a*3:
				return a

	return 0

def evaluatePos(board: np.array, pos: np.array, square: int, symbol: int, condition_1 = False):

	pos[square] = symbol
	evaluation = 0
	evaluation += points[square]

	for line in line_win:
		if np.sum(pos[line]) == 2 * symbol:
			evaluation += 1
			break

	for line in line_win:
		if np.sum(pos[line]) == 3 * symbol:
			evaluation += 5
			break

	pos[square] = -symbol
	for line in line_win:
		if np.sum(pos[line]) == -3*symbol:
			evaluation += 2
			break

	if condition_1:
		if checkWinCondition(board[square]) or np.all(board[square]):
			evaluation -= 5

	pos[square] = symbol
	evaluation += checkWinCondition(pos)*15*symbol
	pos[square] = 0

	return evaluation


def evaluateGame(board, currentBoard, symbol: int):
	evale = 0
	mainBoard = []

	for eh in range(9):
		evale += realEvaluateSquare(board[eh], symbol)*1.5*evaluatorMul[eh]
		if eh == currentBoard:
			evale += realEvaluateSquare(board[eh], symbol)*evaluatorMul[eh]

		tmpEv = checkWinCondition(board[eh])
		evale += tmpEv*evaluatorMul[eh]*symbol
		mainBoard.append(tmpEv)

	mainBoard = np.array(mainBoard)

	evale += checkWinCondition(mainBoard)*5000*symbol
	evale += realEvaluateSquare(mainBoard, symbol)*150

	return evale

def realEvaluateSquare(pos: np.array, symbol: int):

	evaluation = 0
	if checkWinCondition(pos):
		evaluation += checkWinCondition(pos)*12*symbol
		return evaluation

	for i in range(len(pos)):
		evaluation += pos[i] * points[i] * symbol

	for a in [1, -1]:
		for line in [[0, 1, 2], [3,4,5], [6,7,8]]:
		  if np.sum(pos[line]) == 2 * a:
			evaluation += 6 * a * symbol
			break

		for line in [[0, 3,6], [1,4,7], [2,5,8]]:
		  if np.sum(pos[line]) == 2 * a:
			evaluation += 6 * a * symbol
			break

		for line in [[0, 4, 8], [2, 4, 6]]:
		  if np.sum(pos[line]) == 2 * a:
			evaluation += 7 * a * symbol
			break

	for a in [1, -1]:
		if  (pos[0] + pos[1] == 2*a and pos[2] == -a) or (pos[1] + pos[2] == 2*a and pos[0] == -a) or (pos[0] + pos[2] == 2*a and pos[1] == -a) or \
			(pos[3] + pos[4] == 2*a and pos[5] == -a) or (pos[3] + pos[5] == 2*a and pos[4] == -a) or (pos[5] + pos[4] == 2*a and pos[3] == -a) or \
			(pos[6] + pos[7] == 2*a and pos[8] == -a) or (pos[6] + pos[8] == 2*a and pos[7] == -a) or (pos[7] + pos[8] == 2*a and pos[6] == -a) or \
			(pos[0] + pos[3] == 2*a and pos[6] == -a) or (pos[0] + pos[6] == 2*a and pos[3] == -a) or (pos[3] + pos[6] == 2*a and pos[0] == -a) or \
			(pos[1] + pos[4] == 2*a and pos[7] == -a) or (pos[1] + pos[7] == 2*a and pos[4] == -a) or (pos[4] + pos[7] == 2*a and pos[1] == -a) or \
			(pos[2] + pos[5] == 2*a and pos[8] == -a) or (pos[2] + pos[8] == 2*a and pos[5] == -a) or (pos[5] + pos[8] == 2*a and pos[2] == -a) or \
			(pos[0] + pos[4] == 2*a and pos[8] == -a) or (pos[0] + pos[8] == 2*a and pos[4] == -a) or (pos[4] + pos[8] == 2*a and pos[0] == -a) or \
			(pos[2] + pos[4] == 2*a and pos[6] == -a) or (pos[2] + pos[6] == 2*a and pos[4] == -a) or (pos[4] + pos[6] == 2*a and pos[2] == -a):

			evaluation -= 9 * a * symbol


	return evaluation

def find_max_curr_board(board, symbol):
	best_idx, best_val = 0, -np.inf

	for curr_board in range(9):
		if checkWinCondition(board[curr_board]) == 0 and np.sum(board[curr_board] == 0) != 0:
			val = realEvaluateSquare(board[curr_board], symbol)
			if best_val < val:
				best_val = val
				best_idx = curr_board

	print(f"Best index and value:  {best_idx} and  {best_val}")
	return best_idx

def miniMax(board, curr_board: int, depth: int, alpha, beta, maximizingPlayer, symbol, RUNS: int):

	RUNS += 1
	tempPos = -1
	Eval = evaluateGame(board, curr_board, symbol)

	if depth <= 0 or np.abs(Eval) > 5000:
		return {"max_eval": Eval, "temp_pos": tempPos}

	if curr_board != -1:
		if checkWinCondition(board[curr_board]) != 0 or np.sum(board[curr_board] == 0) == 0:
			curr_board = -1

	if maximizingPlayer:
		maxEval = -np.inf
		evalua = {"max_eval": -np.inf, "temp_pos": tempPos}

		for idx_board in range(9):
			if curr_board == -1:
				for trr in range(9):
					if checkWinCondition(board[idx_board]) == 0:
						if board[idx_board][trr] == 0:
							board[idx_board][trr] = symbol
							evalua = miniMax(board, trr, depth-1, alpha, beta, False, symbol,RUNS)
							board[idx_board][trr] = 0

							if evalua['max_eval'] > maxEval :
								maxEval = evalua['max_eval']
								tempPos = idx_board

							alpha = max(alpha, evalua['max_eval'])

							if beta <= alpha:
								break

			else:
				if board[curr_board][idx_board] == 0:
					board[curr_board][idx_board] = symbol
					evalua = miniMax(board, idx_board, depth-1, alpha, beta, False, symbol, RUNS)
					board[curr_board][idx_board] = 0

					if evalua['max_eval'] > maxEval:
						maxEval = evalua['max_eval']
						tempPos = evalua['temp_pos']

					alpha = max(alpha, evalua['max_eval'])

					if beta <= alpha:
						break

		return {"max_eval": maxEval, "temp_pos": tempPos}

	else:

		minEval = np.inf
		evalua = {"max_eval": np.inf, "temp_pos": tempPos}

		for idx_board in range(9):
			if curr_board == -1:
				for trr in range(9):
					if checkWinCondition(board[idx_board]) == 0:
						if board[idx_board][trr] == 0:
							board[idx_board][trr] = -symbol
							evalua = miniMax(board, trr, depth-1, alpha, beta, True, symbol, RUNS)
							board[idx_board][trr] = 0

							if evalua['max_eval'] < minEval:
								minEval = evalua['max_eval']
								tempPos = idx_board

							beta = min(beta, evalua['max_eval'])

							if beta <= alpha:
								break

			else:
				if board[curr_board][idx_board] == 0:
					board[curr_board][idx_board] = -symbol
					evalua = miniMax(board, idx_board, depth-1, alpha, beta, True, symbol, RUNS)
					board[curr_board][idx_board] = 0

					if evalua['max_eval'] < minEval:
						minEval = evalua['max_eval']
						tempPos = evalua['temp_pos']

					beta = min(beta, evalua['max_eval'])
					if beta <= alpha:
						break

		return {"max_eval": minEval, "temp_pos": tempPos}


def count_empty_square(board: np.array):
	count = 0
	for curr_board in range(board.shape[0]):
		if checkWinCondition(board[curr_board]) == 0:
			count += np.sum(board[curr_board] == 0)

	return count

def simulate(board, currentBoard, moves, symbol, depth = 1, condition_1 = False):

	bestMove = -1
	bestScore = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])

	empty_squares = count_empty_square(board)

	if currentBoard == -1 or checkWinCondition(board[currentBoard]) != 0:
		#savedMm = None
		savedMm = miniMax(board, -1, min(2, empty_squares), -np.inf, np.inf, True, symbol, RUNS)
		currentBoard = savedMm['temp_pos']

	for idx in range(9):
		if board[currentBoard][idx] == 0:
			bestMove = idx
			break

	if bestMove != -1:
		for idx in range(9):
			if board[currentBoard][idx] == 0:
				score = evaluatePos(board, board[currentBoard], idx, symbol, condition_1=condition_1)*45
				bestScore[idx] = score

		for idx in range(9):
			if checkWinCondition(board[currentBoard]) == 0:
				if (board[currentBoard][idx] == 0):
					board[currentBoard][idx] = symbol
					savedMm = None

					if(moves >= 20):
						savedMm = miniMax(board, idx, min(2, empty_squares), -np.inf, np.inf, False, symbol, RUNS)
					else:
						savedMm = miniMax(board, idx, min(depth, empty_squares), -np.inf, np.inf, False, symbol, RUNS)

					score2 = savedMm['max_eval']
					board[currentBoard][idx] = 0
					bestScore[idx] += score2

		for idx in range(len(bestScore)):
			if(bestScore[idx] > bestScore[bestMove]):
				bestMove = idx

		return currentBoard, int(bestMove //3), int(bestMove % 3), symbol
	else:
		return None


def chooseSquare(board, currentBoard):
	pass


