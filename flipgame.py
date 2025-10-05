import random

def generate_possible_next_moves(s: str) -> list:
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
        board_string = ''.join(random.choices(['+', '-'], k=lenght))
        if board_string.count("++") > 2:
            return board_string
            
def get_player_move(board_state: str, possible_moves: list) -> str:
    ''' 
    asks the player for the move and returns the board state after

    Args:
        board_state (str): current board state
        possible_moves (list): list of moves the player can make in this turn 

    Returns:
        str: board state after the player made his move
    '''

    # display user optiones 
    print("Possible next moves:")
    for i, move in enumerate(possible_moves):
        print(f"{i} - [{move}]")

    while True:
        player_input = -1
        try:
            player_input = int(input("Enter the number corresponding to your choice: "))
        except ValueError:
            print("Invalid value. Try a number")
            continue

        if  0 <= player_input < len(possible_moves):
            print("Valid move!")
            return possible_moves[player_input]
        else:
            print("Invalid move.") 


if __name__ == "__main__":
    board_state = generate_board(10)
    
    while True:
        print(board_state)
        possible_moves = generate_possible_next_moves(board_state)
        print("Game board:", board_state)
        if not possible_moves:
            print("Player you won!")
            break
        board_state = get_player_move(board_state, possible_moves)

    print("Game over.")
