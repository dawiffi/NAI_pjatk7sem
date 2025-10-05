def generate_possible_next_moves(s: str):
    """
    Generate all possible states of the string after one valid move.
    A move is flipping two consecutive "++" into "--".
    """
    result = []
    for i in range(len(s) - 1):
        if s[i] == '+' and s[i + 1] == '+':
            result.append(s[:i] + '--' + s[i + 2:])
    return result


if __name__ == "__main__":
    import random
    length = random.randint(4, 10)
    while True:
        input_string = ''.join(random.choice(['+', '-']) for _ in range(length))
        if '++' in input_string:
            break
    input_string = ''.join(random.choice(['+', '-']) for _ in range(length))

    print("Input:", input_string)
    print("Possible next moves:", generate_possible_next_moves(input_string))

    player_input = input("Enter your move (e.g., '++' to '--'): ")
    if player_input in generate_possible_next_moves(input_string):
        print("Valid move!")
        new_state = player_input
        current_player = 1
        while True:
            print(f"Current player: {current_player}")
            current_player = 2 if current_player == 1 else 1
            player_input = input(f"Player {current_player}, enter your move (e.g., '++' to '--'): ")
            if player_input in generate_possible_next_moves(new_state):
                print("Valid move!")
                new_state = player_input
                current_player = 1
                while True:
                    if not generate_possible_next_moves(new_state):
                        print(f"Player {current_player} wins!")
                        break
                    print(f"Current player: {current_player}")
                    current_player = 2 if current_player == 1 else 1
                    player_input = input(f"Player {current_player}, enter your move (e.g., '++' to '--'): ")
                    if player_input in generate_possible_next_moves(new_state):
                        print("Valid move!")
                        new_state = player_input
                    else:
                        print("Invalid move.")
        print("Game over.")
    else:
        print("Invalid move.")
