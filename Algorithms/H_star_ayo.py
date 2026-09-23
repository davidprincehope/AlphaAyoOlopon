def evaluate_state(state_view, player_id):
    board = state_view.board
    captured = state_view.captured
    other = 1 - player_id
    own = board[player_id * 6:player_id * 6 + 6]
    opp = board[other * 6:other * 6 + 6]
    seed_diff = sum(own) - sum(opp)
    total = sum(own) + sum(opp)
    mobility_diff = (own[0] > 0) + (own[1] > 0) + (own[2] > 0) + (own[3] > 0) + (own[4] > 0) + (own[5] > 0) - (opp[0] > 0) - (opp[1] > 0) - (opp[2] > 0) - (opp[3] > 0) - (opp[4] > 0) - (opp[5] > 0)
    return max(-10.0, min(10.0, 0.80 * (captured[player_id] - captured[other]) + 0.18 * seed_diff / (1 + total) + 0.30 * mobility_diff))
