"""
Tic Tac Toe Player
"""

import math

#Constants
X_WIN_VALUE = 1
O_WIN_VALUE = -1
X = "X"
O = "O"
EMPTY = None
INF = math.inf

#variables
current_player = ""

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    o_counter = 0
    x_counter = 0
    global current_player
    if not terminal(board):
        for row in board:
            for elem in row:
                if ( elem is X):   
                    x_counter+=1
                elif ( elem is O):
                    o_counter+=1
                else:
                    pass
        if ( x_counter <= o_counter):
            current_player = X
            return X
        elif ( x_counter == 0 and o_counter == 0):
            current_player = X
            return X
        else :
            current_player = O
            return O



def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if ( board[i][j] == EMPTY):
                actions.append( (i,j) )
    return actions

def result(board, action : (int, int)):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    board[ action[0] ][ action[1] ] = current_player   
    return board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    game_result = utility(board)
    if game_result == X_WIN_VALUE:
        return X
    elif game_result == O_WIN_VALUE:
        return O
    elif game_result == 0: 
        return None
    else:
        raise RuntimeError("Incorrect winner function behaviour.")

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    game_vale = utility(board)
    if (game_vale == None):
        return False
    else:
        return True

def _isBoardFull(board):
    for row in board:
        for element in row:
            if element == EMPTY:
                return False
    return True

def utility(board) -> int:
    """
    Returns 1 if X has won the game, -1 if O has won, 0 if tie, None if game did not end yet.
    """
    winner = None
    #"""Diagonal Case"""
    if board[0][0] == board[1][1] and board[0][0] == board[2][2] and board[0][0] != EMPTY:
        winner = board[0][0]
    if board[0][2] == board[1][1] and board[0][2] == board[2][0] and board[0][2] != EMPTY:
        winner = board[0][2]
    #"""Case Row Horizontal"""
    for i in range(len(board)):
        if board[i][0] == board[i][1] and board[i][0]== board[i][2] and board[i][0] != EMPTY:
            winner = board[i][0]
    # """Case Row Vertical"""
    for i in range(len(board)):
        all_in_vertical = True
        for j in range(1, len(board[i])):
            if board[0][i] != board[j][i]:
                all_in_vertical = False
        if all_in_vertical:
            winner = board[0][i]
    
    if winner == None:
        if _isBoardFull(board):
            return 0
        else:
            return None
    elif winner == X:
        return X_WIN_VALUE
    elif winner == O:
        return O_WIN_VALUE

def minimax(board) -> (int, int):
    """
    Returns the optimal action for the current player on the board.
    """
    if current_player == O:
        best_score = INF
        nextIsMaxPlayer = True
    else:
        best_score = -INF
        nextIsMaxPlayer = False
    best_move = None

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                board[i][j] = current_player
                score = _minimax_process(board, 0, nextIsMaxPlayer)
                board[i][j] = EMPTY
                if compareTakingPlayer(score, best_score, current_player):
                    best_score = score
                    best_move = (i,j)
    return best_move

def compareTakingPlayer(score, b_score, player):
    if player == O:
        return score < b_score
    else:
        return score > b_score


def _minimax_process(board, depth, isMaximizingPlayer):
    result = utility(board)
    if result != None:
        score = result
        return score
    if isMaximizingPlayer:
        best_score_max_player = -INF
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == EMPTY:
                    board[i][j] = X
                    score = _minimax_process(board, depth+1, False)
                    board[i][j] = EMPTY
                    best_score_max_player = max(score, best_score_max_player)
        return best_score_max_player
    else:
        best_score_min_player = INF
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == EMPTY:
                    board[i][j] = O
                    score = _minimax_process(board, depth+1, True)
                    board[i][j] = EMPTY
                    best_score_min_player = min(score, best_score_min_player)
        return best_score_min_player
