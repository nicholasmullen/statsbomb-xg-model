import warnings
import numpy as np
import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")


def pull_shots(competition_id, season_id, cache_path=None):
    """Download every shot for a competition-season. Optionally cache to a pickle."""
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    frames = []
    for i, match_id in enumerate(matches.match_id, start=1):
        try:
            ev = sb.events(match_id=match_id)
        except Exception as e:
            print(f"Skipping match {match_id}: {e}")
            continue
        s = ev[ev.type == "Shot"].copy()
        s["match_id"] = match_id
        frames.append(s)
        if i % 25 == 0:
            print(f"{i}/{len(matches)} matches done")
    shots = pd.concat(frames, ignore_index=True)
    if cache_path:
        shots.to_pickle(cache_path)
    return shots


def _in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _shot_angle(x, y):
    left = np.array([120 - x, 36 - y])
    right = np.array([120 - x, 44 - y])
    cos_a = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    return np.arccos(np.clip(cos_a, -1, 1))


def _freeze_features(row):
    ff = row.shot_freeze_frame
    empty = pd.Series({"defenders_in_cone": np.nan, "gk_dist_to_goal": np.nan, "nearest_defender": np.nan})
    if not isinstance(ff, list):
        return empty

    shooter = (row.x, row.y)
    defenders = [f["location"] for f in ff
                 if not f["teammate"] and f["position"]["name"] != "Goalkeeper"]
    keepers = [f["location"] for f in ff
               if not f["teammate"] and f["position"]["name"] == "Goalkeeper"]

    in_cone = sum(_in_triangle(d, shooter, (120, 36), (120, 44)) for d in defenders)
    gk_dist = np.hypot(120 - keepers[0][0], 40 - keepers[0][1]) if keepers else np.nan
    nearest = min((np.hypot(d[0] - row.x, d[1] - row.y) for d in defenders), default=np.nan)
    return pd.Series({"defenders_in_cone": in_cone, "gk_dist_to_goal": gk_dist, "nearest_defender": nearest})


def build_features(shots):
    """Turn raw StatsBomb shots into a model-ready table (penalties and shootouts removed)."""
    shots = shots[(shots.shot_type != "Penalty") & (shots.period != 5)].copy()
    shots["x"] = shots.location.apply(lambda loc: loc[0])
    shots["y"] = shots.location.apply(lambda loc: loc[1])
    shots["distance"] = np.sqrt((120 - shots.x) ** 2 + (40 - shots.y) ** 2)
    shots["angle"] = shots.apply(lambda r: _shot_angle(r.x, r.y), axis=1)
    shots["goal"] = (shots.shot_outcome == "Goal").astype(int)
    shots["is_header"] = (shots.shot_body_part == "Head").astype(int)
    shots = shots.join(shots.apply(_freeze_features, axis=1))

    cols = ["match_id", "player", "team", "x", "y", "distance", "angle", "is_header",
            "shot_type", "goal", "shot_statsbomb_xg",
            "defenders_in_cone", "gk_dist_to_goal", "nearest_defender"]
    return shots[cols].reset_index(drop=True)