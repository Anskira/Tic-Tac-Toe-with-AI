"""

Name: Anshul Omkar Patil
UFID: 13265377

"""

import numpy as np
import joblib
import sklearn


# Loading the ML model that te player will play against
model = joblib.load(
    "D:/Programming for applied data science/Mini-project2/tic_tac_toe_rf_model.joblib"
)


# This class stores the state of the board at any given time in the game
class Board:
    def __init__(self):
        self.c = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]

    def printBoard(self):
        BOARD_HEADER = "-----------------\n|R\\C| 0 | 1 | 2 |\n-----------------"
        print(BOARD_HEADER)
        for row in range(len(self.c)):
            print(f"| {row} |", end="")
            for i in range(len(self.c)):
                print(f" {self.c[row][i]:1} |", end="")
            print("")
            print("------------------")


# this class contains all the methods that help operate the game
class Game:

    def __init__(self):
        self.board = Board()
        self.turn = "X"

    # method to switch players
    def switchPlayer(self, count):
        if count % 2 == 1:
            self.turn = "O"
        else:
            self.turn = "X"

    # method to validate user entry
    def validateEntry(self, row, col):
        board = self.board.c
        r, c = row, col
        if r > 2 or r < 0 or c > 2 or c < 0:
            print("Invalid entry: try again.")
            print("Row & column numbers must be either 0, 1, or 2.")
            return False
        if board[r][c] != " ":
            print("The cell is already taken.\nPlease make another selection")
            return False
        return True

    # method to check if the board is full
    def checkFull(self):
        list1 = self.board.c

        # nested for loop to check whether the board is full
        for row in range(len(list1)):
            for col in range(len(list1)):
                if list1[row][col] == " ":
                    return False  # returns false if the board has even a single empty space available

        return (
            True  # returns false if the board has even a single empty space available
        )

    # this method checks for a winner
    def checkWin(self):

        list1 = self.board.c

        # nested for loop to iterate through the rows and check if any rows satisfy the condition for a win
        for row in list1:
            count = 0
            for i in row:
                if i != " ":
                    count += 1

            if count == 3 and len(set(row)) == 1:  # winning condition
                return True  # returns true if winning condition is satisfied

        # nested for loop to iterate through the columns(by taking transpose of the list) and check if any columns satisfy the condition for a win
        for row in np.transpose(list1):
            count = 0
            for i in row:
                if i != " ":
                    count += 1

            if count == 3 and len(set(row)) == 1:  # winning condition
                return True  # returns true if winning condition is satisfied

        # storing the first diagonal values in a list and checking if the first diagonal satisfies the condition for a win
        diagonal_1 = [list1[i][i] for i in range(len(list1))]
        diagonal_1_count = 0

        for i in diagonal_1:
            if i != " ":
                diagonal_1_count += 1

        if diagonal_1_count == 3 and len(set(diagonal_1)) == 1:  # winning condition
            return True  # returns true if winning condition is satisfied

        # storing the second diagonal values in a list and checking if the second diagonal satisfies the condition for a win
        diagonal_2 = [list1[i][len(list1) - i - 1] for i in range(len(list1))]
        diagonal_2_count = 0

        for i in diagonal_2:
            if i != " ":
                diagonal_2_count += 1

        if diagonal_2_count == 3 and len(set(diagonal_2)) == 1:  # winning condition
            return True  # returns true if winning condition is satisfied

        # returns false if no winning condition is satisfied
        return False

    # this method checks if the game has met an end condition by calling checkFull() and checkWin()
    # hint: you can call a class method using self.method_name() within another class method, e.g., self.checkFull()
    def checkEnd(self):
        # print("here")
        win = self.checkWin()
        full = self.checkFull()
        if win == True:
            print("Thank you for your selection.")
            print(f"{self.turn} IS THE WINNER!!!")
            self.board.printBoard()

        # calling the function to check if the board is full
        elif full == True:
            print("Thank you for your selection.")
            print("")
            print("DRAW! NOBODY WINS!")
            self.board.printBoard()

        if win == True or full == True:
            return True

        else:
            return False

    # this method is used for AI's move based on the model prediction
    def ai_move(self):
        board_state = [
            1 if cell == "X" else -1 if cell == "O" else 0
            for row in self.board.c
            for cell in row
        ]
        best_move_index = model.predict([board_state])[0]
        row, col = divmod(best_move_index, 3)
        if self.validateEntry(row, col):
            self.board.c[row][col] = self.turn
            print(f"\nAI places O at position: ({row}, {col})")

    # this method runs the tic-tac-toe game
    # hint: you can call a class method using self.method_name() within another class method
    def playGame(self):
        print("New Game: X goes first.\n")
        self.board.printBoard()
        count = 0
        while True:
            if self.turn == "X":
                valid = False
                # storing the row and column number in variables r and c
                while not valid:
                    print(f"\n{self.turn}'s turn.")
                    print(f"Where do you want your {self.turn} placed?")
                    r, c = map(
                        int,
                        input(
                            "Please enter row number and column number separated by a comma.\n"
                        ).split(","),
                    )
                    valid = self.validateEntry(r, c)
                self.board.c[r][c] = self.turn
            else:
                print(f"\n{self.turn}'s (AI) turn.")
                self.ai_move()
            end = self.checkEnd()
            if end == True:
                break
            print("Thank you for your selection.")
            self.board.printBoard()
            count += 1
            self.switchPlayer(count)


# main function
def main():

    game = Game()
    game.playGame()
    # first initializes a variable to repeat the game
    repeat = input(
        "\nAnother game? Enter Y or y for yes.\n"
    )  # asking players if they want to play the game again
    if repeat[0].lower() == "y":
        main()

    # goodbye message
    print("Thank you for playing!")


# call to main() function
if __name__ == "__main__":
    main()
