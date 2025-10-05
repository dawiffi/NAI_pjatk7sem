import random

def generate_possible_next_moves(s: str):
    """
    Generate all possible states of the string after one valid move.
    A move is flipping two consecutive "++" into "--".

    Args:
        s (str): string representing current board state e.g. "++--+-++-"

    Returns:
        list: list of strings containing currently posible valid moves
    """
    result = []
    for i in range(len(s) - 1):
        if s[i] == '+' and s[i + 1] == '+':
            result.append(s[:i] + '--' + s[i + 2:])
    return result

def generate_board(lenght : int) -> str:
    ''' 
    Creates a random string made out of '+' and '-' chars of a given length
    this is our game board.

    Args:
        length (int): Length of the game board 

    Returns:
        str: the random level used for the game e.g. "++--+-++-"
    '''
    board_string = ''
    while True:
        board_string = ''.join(random.choices(['+', '-'], k=lenght) )
        if board_string.count("++") > 2:
            return board_string

if __name__ == "__main__":
    board_state = generate_board(10)

    while True:
        posible_moves = generate_possible_next_moves(board_state)
        print("Game board:", board_state)
        print("Possible next moves:", posible_moves, not posible_moves)
        if not posible_moves:
            print("Player you won!")
            break
        
        player_input = input("enter your move (e.g., '++' to '--'): ")
        if player_input in posible_moves:
            print("Valid move!")
            board_state = player_input
        else:
            print("Invalid move.")
            continue 
    print("Game over.")
