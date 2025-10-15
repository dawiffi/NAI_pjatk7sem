import random
from anytree import Node, RenderTree

# maximal amount of steps when generating tree
MAX_SEARCH_DEPTH = 8 
# max length of the board (string size)
MAX_BOARD_SIZE = 12 
# minimal amout of "++" occurrences for the board to be considered playable
# this is so we don't generate game boards that are solved before the game begins
MINIMAL_DOUBLE_SIGHN_OCCURANCE = 3 

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
    Returns:
        int: score of the branch needed for ai to make the final decision 
    '''
    parent_board = parent.name
    next_moves = generate_possible_next_moves(parent_board)
    if not next_moves or parent.depth > MAX_SEARCH_DEPTH:
        parent.is_last_child = True
        if parent.depth % 2 == 0: # player wins on even ai wins on odd 
            return -5 
        else:
            return 1
    else:
        branch_value = 0 #branch_value is the sum of its children values 
        for move in next_moves:
            new_child = Node(move, parent=parent, depth=parent.depth+1, is_last_child=False, score=0)
            branch_value += create_next_branches(new_child)
        parent.score = branch_value
    return parent.score

def visualise_node_data(parent: Node):
    '''
    visualises all the generated branches of the given node 

    Args:
        parent (Node): the parent node we want to display all children of 
    '''
    for pre, fill, node in RenderTree(origial):
        if node.is_last_child:
            if node.depth % 2 == 0:
                print("%s\033[91m%s\033[00m" % (pre, node.name)) # player wins on even turns so we print it red
            else:
                print("%s\033[92m%s\033[00m" % (pre, node.name)) # ai wins on odd turns so we print green
        else:
            print("%s%s %s" % (pre, node.name, node.score))


def player_turn(board: str) -> str:
    '''
    Processes all the logic of player turn.

    Args:
        board (str): current board state
    Returns:
        str: board state after player turn 
    '''
    # inform player
    possible_moves = generate_possible_next_moves(board_state)
    print("Game board:", board_state)

    #player move
    return get_player_move(board_state, possible_moves)

def ai_turn(board: str) -> str:
    '''
    Processes all the logic of ai turn.

    Args:
        board (str): current board state
    Returns:
        str: board state after ai turn 
    '''
    # ai turn 
    origial = Node(board_state, depth=0, is_last_child=False)
    create_next_branches(origial)
    #visualise_node_data(origial) #uncomment this to see visualisation of how the ai thinks

    #select move with the highest score 
    favorite_child = origial.children[0]
    for potential_child in origial.children:
        if potential_child.score > favorite_child.score:
            favorite_child = potential_child
    
    #return move
    print(f"Ai moved: {board_state}")
    return favorite_child.name

def chceck_win(win_text: str) -> bool:
    '''
    chceck the win condition and return it. also print the given win text

    Args:
        win_text (str): text that will be printed if the game ended
    Returns:
        str: returns if the game has ended 
    '''
    #check win condition
    possible_moves = generate_possible_next_moves(board_state)

    if not possible_moves:
        print(win_text)
        return True
    return False

if __name__ == "__main__":
    board_state = generate_board(MAX_BOARD_SIZE)

    while True:
        board_state = player_turn(board_state)
        if chceck_win("Player won!"):
            break

        board_state = ai_turn(board_state)
        if chceck_win("Ai won!"):
            break

    print("Game over.")
