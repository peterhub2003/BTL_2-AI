import numpy as np
from importlib import import_module
from state import UltimateTTT_Move

evaluate = import_module("Evaluator")

def count_moves(cur_state, player_to_move):
    board = np.array([block.flatten() for block in cur_state.blocks]).flatten()
    return np.sum(board == player_to_move)


def utils(cur_state):
    if cur_state.previous_move != None:
        index_local_board = cur_state.previous_move.x * 3 + cur_state.previous_move.y
    else:
        index_local_board = -1

    board =  np.array([block.flatten() for block in cur_state.blocks])
    return board, index_local_board


def select_move(cur_state, remain_time):

    board, currentBoard = utils(cur_state)

    m = cur_state.get_valid_moves
    if len(m) == 0:
        return None

    if currentBoard == -1:
        index_local_board, x, y, value = 4, 1, 1, cur_state.player_to_move
    else:
        MOVES = count_moves(cur_state, cur_state.player_to_move)
        # print(MOVES)
        index_local_board, x, y, value = evaluate.simulate(board, currentBoard, moves = MOVES, symbol = cur_state.player_to_move, depth = 2, condition_1 = True)

    valid_move = UltimateTTT_Move(index_local_board, x, y, value)
    if valid_move != None:
        return valid_move

    return None
