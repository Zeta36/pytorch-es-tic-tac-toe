from __future__ import absolute_import, division, print_function

import numpy as np
import random


class TicTacToeEnv:

    def __init__(self):
        self.observation_space = 9
        self.board = list('---------')
        self.action_space = 9

    def step(self, action):
        """
        Given an action, the model moves and we observe the new state and return the reward .
        Arguments:
              action: a integer between 0 and 8 (the place to move).
        Return Value:
               new state of board, reward, game ended (True|False)
        """
        if self.board[action] != '-':
            # Ilegal move
            return self.getBoard(), -0.2, True

        #Model moves
        self.board[action] = 'X'

        #Is Draw
        if self.isDraw(self.board):
            return self.getBoard(), 1, True

        #Classic AI moves
        reward, move = self.nextMove(self.board, 'O')
        self.board[move] = 'O'

        # Classic AI wins
        if self.isWin(self.board):
            return self.getBoard(), -1, True

        #Is Draw
        if self.isDraw(self.board):
            return self.getBoard(), 1, True

        return self.getBoard(), 0.1, False

    def getBoard(self):
        board = np.asarray([ord(i) for i in self.board], np.float32)
        return board

    def clearProb(self, prob):
        for i in range(9):
            if self.board[i] != '-':
                prob[i] = 0.0
        return prob

    def chunks(slef, l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    def reset(self, first):
        """
        Start a new game where first move 'first'
        Return: the state of the reset board
        """
        self.board = list('---------')
        if first == 0:
            p = random.random()
            if p < 0.5:
                _, move = self.nextMove(self.board, 'O')
                self.board[move] = 'O'
                print("\nClassical AI moves first with 'O':\n")
                return self.getBoard()
            else:
                print("\nModel moves first with 'X':\n")
                return self.getBoard()
        elif first == 1:
            return self.getBoard()
        else:
            _, move = self.nextMove(self.board, 'O')
            self.board[move] = 'O'
            return self.getBoard()

    def render(self):
        """
        Print the board state
        """
        board2D = np.array(self.chunks(self.board, 3))
        for i in range(3):
                print(board2D[i].flatten())

        print('\nX = Model AI, O = Classical AI\n')

    def isDraw(self, board):
        """
        Was the game a draw?        
        Return: True|False
        """
        for i in range(9):
            if board[i] == '-':
                return False
        return True

    # Taken from https://gist.github.com/SudhagarS/3942029
    def isWin(self, board):
        """
        Given a board checks if it is in a winning state.
        Arguments:
              board: a list containing X,O or -.
        Return Value:
               True if board in winning state. Else False
        """
        ### check if any of the rows has winning combination
        for i in range(3):
            if len(set(board[i * 3:i * 3 + 3])) is 1 and board[i * 3] is not '-': return True
        ### check if any of the Columns has winning combination
        for i in range(3):
            if (board[i] is board[i + 3]) and (board[i] is board[i + 6]) and board[i] is not '-':
                return True
        ### 2,4,6 and 0,4,8 cases
        if board[0] is board[4] and board[4] is board[8] and board[4] is not '-':
            return True
        if board[2] is board[4] and board[4] is board[6] and board[4] is not '-':
            return True
        return False

    def nextMove(self, board, player):
        """
        Computes the next move for a player given the current board state and also
        computes if the player will win or not.
        Arguments:
            board: list containing X,- and O
            player: one character string 'X' or 'O'
        Return Value:
            willwin: 1 if 'X' is in winning state, 0 if the game is draw and -1 if 'O' is
                        winning
            nextmove: position where the player can play the next move so that the
                             player wins or draws or delays the loss
        """
        ### when board is '---------' evaluating next move takes some time since
        ### the tree has 9! nodes. But it is clear in that state, the result is a draw
        if len(set(board)) == 1: return 0, 4

        nextplayer = 'X' if player == 'O' else 'O'
        if self.isWin(board):
            if player is 'X':
                return -1, -1
            else:
                return 1, -1
        res_list = []  # list for appending the result
        c = board.count('-')
        if c is 0:
            return 0, -1
        _list = []  # list for storing the indexes where '-' appears
        for i in range(len(board)):
            if board[i] == '-':
                _list.append(i)
        # tempboardlist=list(board)
        for i in _list:
            board[i] = player
            ret, move = self.nextMove(board, nextplayer)
            res_list.append(ret)
            board[i] = '-'
        if player is 'X':
            maxele = max(res_list)
            return maxele, _list[res_list.index(maxele)]
        else:
            minele = min(res_list)
            return minele, _list[res_list.index(minele)]