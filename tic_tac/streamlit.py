# Tic Tac Toe using Agentic AI (Minimax)

def print_board(board):
    for i in range(0, 9, 3):
        print(board[i], "|", board[i+1], "|", board[i+2])
    print()

def check_winner(board):
    win_positions = [
        [0,1,2],[3,4,5],[6,7,8],   # rows
        [0,3,6],[1,4,7],[2,5,8],   # columns
        [0,4,8],[2,4,6]            # diagonals
    ]
    
    for pos in win_positions:
        if board[pos[0]] == board[pos[1]] == board[pos[2]] != " ":
            return board[pos[0]]
    return None

def is_draw(board):
    return " " not in board

# 🎯 Agent (Minimax AI)
def minimax(board, is_maximizing):
    winner = check_winner(board)
    
    if winner == "O":   # AI wins
        return 1
    elif winner == "X": # Human wins
        return -1
    elif is_draw(board):
        return 0

    if is_maximizing:
        best_score = -float("inf")
        for i in range(9):
            if board[i] == " ":
                board[i] = "O"
                score = minimax(board, False)
                board[i] = " "
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = float("inf")
        for i in range(9):
            if board[i] == " ":
                board[i] = "X"
                score = minimax(board, True)
                board[i] = " "
                best_score = min(score, best_score)
        return best_score

# 🤖 AI Move (Agent Decision)
def ai_move(board):
    best_score = -float("inf")
    move = -1
    
    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(board, False)
            board[i] = " "
            
            if score > best_score:
                best_score = score
                move = i
                
    return move

# 🎮 Game Loop
def play_game():
    board = [" "] * 9
    
    while True:
        print_board(board)

        # Human move
        try:
            pos = int(input("Enter position (0-8): "))
        except ValueError:
            print("Invalid input! Please enter a number between 0 and 8.")
            continue

        if pos < 0 or pos > 8 or board[pos] != " ":
            print("Invalid move! Try again.")
            continue
            
        board[pos] = "X"

        if check_winner(board):
            print_board(board)
            print("You win!")
            break
        if is_draw(board):
            print_board(board)
            print("Draw!")
            break

        # AI move
        print("AI is thinking...")
        move = ai_move(board)
        board[move] = "O"

        if check_winner(board):
            print_board(board)
            print("AI wins!")
            break
        if is_draw(board):
            print_board(board)
            print("Draw!")
            break

if __name__ == "__main__":
    play_game()
