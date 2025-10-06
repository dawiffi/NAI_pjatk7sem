import random
from anytree import Node, RenderTree

# maximal amount of steps when generating tree
MAX_SEARCH_DEPTH = 8 
# max length of the board (string size)
MAX_BOARD_SIZE = 10 
# minimal amout of "++" occurrences for the board to be considered playable
# this is so we don't generate game boards that are solved before the game begins
MINIMAL_DOUBLE_SIGHN_OCCURANCE = 2 

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
        if board_string.count("++") > MINIMAL_DOUBLE_SIGHN_OCCURANCE:
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

def create_next_branches(parent: Node):
    '''
    calculates and appends next branches of a node containing a board state
    the function will run until there is no more possible moves or the 
    MAX_SEARCH_DEPTH is exceed.

    Args:
        parent (Node): Node containing a board state that will be analised for next possible moves
    '''
    parent_board = parent.name
    next_moves = generate_possible_next_moves(parent_board)
    if not next_moves or parent.depth > MAX_SEARCH_DEPTH:
        return
    else:
        for move in next_moves:
            new_child = Node(move, parent=parent, depth=parent.depth+1)
            create_next_branches(new_child)


if __name__ == "__main__":
    board_state = generate_board(MAX_BOARD_SIZE)
    #testing 
    origial = Node(board_state, depth=0)
    create_next_branches(origial)

    for pre, fill, node in RenderTree(origial):
        print("%s%s" % (pre, node.name))
    
    while True:
        # player turn
        # inform player
        print(board_state)
        possible_moves = generate_possible_next_moves(board_state)
        print("Game board:", board_state)

        #player move
        board_state = get_player_move(board_state, possible_moves)

        #check win condition
        possible_moves = generate_possible_next_moves(board_state)
        if not possible_moves:
            print("Player you won!")
            break
        
        # ai turn 


    print("Game over.")
