"""Evaluate a checkpoint against random play or rollout MCTS, alternating seats."""

import argparse
import json
from pathlib import Path

import numpy as np

from Algorithms.alpha_zero.runtime import use_repository_open_spiel

use_repository_open_spiel()
from Algorithms.alpha_zero.bot import load_bot
from open_spiel.python.algorithms import mcts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=int, required=True)
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--simulations', type=int, default=100)
    parser.add_argument('--opponent', choices=['random', 'mcts'], default='random')
    parser.add_argument('--opponent-simulations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error('--games must be a positive even number for balanced seats')
    if args.simulations < 2 or args.opponent_simulations < 2:
        parser.error('search simulation counts must be >= 2')
    game, bot = load_bot(args.run, args.checkpoint, args.simulations, args.seed)
    rng = np.random.RandomState(args.seed)
    opponent = mcts.MCTSBot(
        game, 1.414, args.opponent_simulations,
        mcts.RandomRolloutEvaluator(random_state=rng), solve=False, random_state=rng,
    ) if args.opponent == 'mcts' else None
    outcomes, lengths, truncated = [], [], 0
    by_seat = [[], []]
    for index in range(args.games):
        az_player = index % 2
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.current_player() == az_player:
                action = bot.step(state)
            elif opponent is None:
                action = int(rng.choice(state.legal_actions()))
            else:
                action = opponent.step(state)
            state.apply_action(action)
        result = state.returns()[az_player]
        outcomes.append(result)
        by_seat[az_player].append(result)
        lengths.append(state.ply)
        truncated += int(state.truncated)
    wins = outcomes.count(1.0)
    draws = outcomes.count(0.0)
    print(json.dumps({
        'run': str(args.run.resolve()), 'checkpoint': args.checkpoint,
        'opponent': args.opponent, 'simulations': args.simulations,
        'opponent_simulations': args.opponent_simulations if opponent else None,
        'seed': args.seed, 'games': args.games, 'wins': wins,
        'draws': draws, 'losses': outcomes.count(-1.0),
        'score_rate': (wins + 0.5 * draws) / args.games,
        'mean_return_by_seat': [float(np.mean(values)) for values in by_seat],
        'mean_moves': float(np.mean(lengths)), 'truncated_games': truncated,
    }, indent=2))


if __name__ == '__main__':
    main()
