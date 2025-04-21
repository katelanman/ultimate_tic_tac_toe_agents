from game import UltimateTicTacToe
from player import Player
from MCTS_player import MCTSPlayer

from tqdm import tqdm
from collections import Counter

import itertools
import time
from multiprocessing import Pool

def mcts_vs_rand(params):
    """pit mcts agent against random player with specified parameters"""
    mcts_id, rand_id, w, t, r, grid_size = params
    
    rand_agent = Player(rand_id)
    mcts_agent = MCTSPlayer(mcts_id, exploration_weight=w, calculation_time=t, n_rollouts=r)
    game = UltimateTicTacToe(rand_agent, mcts_agent, grid_size=grid_size)

    start = time.time()
    result = game.play_game()
    game_time = time.time() - start

    return {
        'result': result, 
        'time': game_time,
        'simulations': mcts_agent.stats["simulations"],
        'plays': mcts_agent.stats["plays"]
    }

def test_mcts(mcts_id, exp_weights, times, rollouts, grid_size=3, n_games=10, verbose=False, n_cores=1):
    all_results = []

    test_params = list(itertools.product(exp_weights, times, rollouts))
    rand_id = 2 if mcts_id == 1 else 1
    for w, t, r in test_params:
        # play n games in parallel
        game_params = [(mcts_id, rand_id, w, t, r, grid_size) for _ in range(n_games)]
        with Pool(processes=n_cores) as pool:
            if verbose:
                game_results = list(tqdm(pool.imap(mcts_vs_rand, game_params), total=n_games))
            else:
                game_results = pool.map(mcts_vs_rand, game_params)
        
        # extract results
        gameplay = [game['result'] for game in game_results]
        times = [game['time'] for game in game_results]
        
        # track stats
        results = Counter(gameplay)
        pct_wins = results[mcts_id]/results.total()
        pct_losses = results[rand_id]/results.total()

        total_simulations = sum(game['simulations'] for game in game_results)
        total_plays = sum(game['plays'] for game in game_results)
        ave_expanded = total_simulations / total_plays if total_plays > 0 else 0

        info = {"exp_weight": w, "calc_time": t, "n_rollouts": r, "pct_wins": pct_wins, 
                "pct_losses": pct_losses, "ave_expanded": ave_expanded, "ave_gametime": sum(times)/len(times)}
        all_results.append(info)

        if verbose:
            print(f"exploration weight: {w}")
            print(f"calculation time: {t}")
            print(f"number of rollouts: {r}")

            print(f"\tpercent wins: {pct_wins}")
            print(f"\tpercent losses: {pct_losses}")
            print(f"\taverage nodes expanded: {ave_expanded}")
            print(f"\taverage gametime: {sum(times)/len(times)}")

    return all_results