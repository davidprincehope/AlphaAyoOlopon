"""Seat advantage in paired fixed-opening matches, independent of agent identity."""


def seat_statistics(records):
    stats = {}
    for seat in (0, 1):
        returns = [record["returns_by_player"][seat] for record in records]
        wins = sum(value > 0 for value in returns)
        draws = sum(value == 0 for value in returns)
        losses = sum(value < 0 for value in returns)
        games = len(returns)
        stats[f"P{seat}"] = {
            "games": games, "wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / games if games else None,
            "draw_rate": draws / games if games else None,
            "loss_rate": losses / games if games else None,
            "score_rate": (wins + 0.5 * draws) / games if games else None}
    return stats


def first_player_advantage(records):
    if not records or any(record.get("opening_plies", 1) % 2 for record in records):
        raise ValueError("This first-player measure requires nonempty even-ply openings (P0 to move)")
    stats = seat_statistics(records)
    return {"definition": "P0 moves first from the saved opening; both agent assignments are included",
            **stats, "p0_score_excess_over_half": stats["P0"]["score_rate"] - 0.5,
            "by_opening_depth": {
                str(depth): seat_statistics([r for r in records if r["opening_plies"] == depth])
                for depth in sorted({r["opening_plies"] for r in records})}}
