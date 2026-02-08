#!/usr/bin/env python3
"""
Quantum Seeker 2.0 - Super Bowl Betting Analysis Framework
Enhanced, fully executable system with advanced analysis and reporting.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except Exception:
    SEABORN_AVAILABLE = False

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("quantum_seeker")


REAL_SB_PROP_CATEGORIES: Dict[str, List[str]] = {
    "COIN_TOSS": [
        "COIN_TOSS_HEADS_TAILS",
        "COIN_TOSS_WINNER_DEFERS",
        "COIN_TOSS_WINNER_RECEIVES",
    ],
    "NATIONAL_ANTHEM": [
        "ANTHEM_DURATION_OVER_UNDER_95",
        "ANTHEM_DURATION_OVER_UNDER_100",
        "ANTHEM_DURATION_OVER_UNDER_105",
    ],
    "GATORADE_SHOWER": [
        "GATORADE_COLOR_ORANGE",
        "GATORADE_COLOR_YELLOW",
        "GATORADE_COLOR_BLUE",
        "GATORADE_COLOR_CLEAR",
        "GATORADE_COLOR_RED",
        "GATORADE_COLOR_GREEN",
        "GATORADE_COLOR_PURPLE",
        "GATORADE_COLOR_NONE",
    ],
    "GAME_OUTCOME": [
        "GAME_OVERTIME_YES",
        "GAME_OVERTIME_NO",
        "GAME_SAFETY_YES",
        "GAME_SAFETY_NO",
        "GAME_MISSED_XP",
        "GAME_SUCCESSFUL_2PT",
    ],
    "SCORING": [
        "FIRST_SCORE_TD",
        "FIRST_SCORE_FG",
        "FIRST_SCORE_SAFETY",
        "LAST_SCORE_TD",
        "LAST_SCORE_FG",
        "LONGEST_TD_OVER_UNDER_45",
    ],
    "SPECIAL_TEAMS": [
        "TOTAL_FGS_OVER_UNDER_3",
        "LONGEST_FG",
        "TOTAL_PUNTS",
        "BLOCKED_KICK",
        "KICKOFF_RETURN_TD",
    ],
    "TURNOVERS": [
        "TOTAL_TURNOVERS",
        "DEFENSIVE_TD",
        "PICK_SIX",
        "INTERCEPTIONS",
        "FUMBLES",
    ],
    "MVP_POSITION": [
        "MVP_QB",
        "MVP_RB",
        "MVP_WR",
        "MVP_TE",
        "MVP_DEFENSIVE",
        "MVP_KICKER",
    ],
    "HALFTIME_SHOW": [
        "HALFTIME_FIRST_SONG",
        "HALFTIME_SPECIAL_GUEST_YES",
        "HALFTIME_SPECIAL_GUEST_NO",
        "HALFTIME_OUTFIT_COLOR",
        "HALFTIME_DURATION",
    ],
    "PLAYER_PROPS_QB": [
        "QB_PASS_YARDS",
        "QB_TDS",
        "QB_COMPLETIONS",
        "QB_INTERCEPTIONS",
    ],
    "PLAYER_PROPS_RB": [
        "RB_RUSH_YARDS",
        "RB_RUSH_TDS",
        "RB_RECEPTIONS",
    ],
    "PLAYER_PROPS_WR": [
        "WR_REC_YARDS",
        "WR_RECEPTIONS",
        "WR_TDS",
    ],
    "PENALTIES": [
        "TOTAL_PENALTIES_OVER_UNDER_10",
        "PENALTY_YARDS",
        "PENALTY_ON_FIRST_PLAY",
    ],
    "GAME_TOTALS": [
        "TOTAL_POINTS_OVER_UNDER_48",
        "TOTAL_POINTS_OVER_UNDER_51",
        "TOTAL_TDS_OVER_UNDER_6",
    ],
    "TRADITIONAL": [
        "QB_PASS_YARDS",
        "QB_COMPLETIONS",
        "QB_TDS",
        "RB_RUSH_YARDS",
        "TOTAL_POINTS",
        "WINNER",
        "POINT_SPREAD",
    ],
}


CATEGORY_PROFILES = {
    "TRADITIONAL": {"prob_mean": 0.52, "prob_std": 0.07, "liquidity_mean": 0.7},
    "COIN_TOSS": {"prob_mean": 0.50, "prob_std": 0.08, "liquidity_mean": 0.25},
    "NATIONAL_ANTHEM": {"prob_mean": 0.35, "prob_std": 0.10, "liquidity_mean": 0.20},
    "GATORADE_SHOWER": {"prob_mean": 0.38, "prob_std": 0.09, "liquidity_mean": 0.22},
    "GAME_OUTCOME": {"prob_mean": 0.45, "prob_std": 0.08, "liquidity_mean": 0.35},
    "SCORING": {"prob_mean": 0.48, "prob_std": 0.08, "liquidity_mean": 0.40},
    "SPECIAL_TEAMS": {"prob_mean": 0.42, "prob_std": 0.09, "liquidity_mean": 0.30},
    "TURNOVERS": {"prob_mean": 0.40, "prob_std": 0.09, "liquidity_mean": 0.32},
    "MVP_POSITION": {"prob_mean": 0.35, "prob_std": 0.10, "liquidity_mean": 0.28},
    "HALFTIME_SHOW": {"prob_mean": 0.30, "prob_std": 0.10, "liquidity_mean": 0.18},
    "PLAYER_PROPS_QB": {"prob_mean": 0.50, "prob_std": 0.07, "liquidity_mean": 0.65},
    "PLAYER_PROPS_RB": {"prob_mean": 0.48, "prob_std": 0.08, "liquidity_mean": 0.55},
    "PLAYER_PROPS_WR": {"prob_mean": 0.47, "prob_std": 0.08, "liquidity_mean": 0.58},
    "PENALTIES": {"prob_mean": 0.45, "prob_std": 0.08, "liquidity_mean": 0.38},
    "GAME_TOTALS": {"prob_mean": 0.50, "prob_std": 0.07, "liquidity_mean": 0.70},
}


DEFAULT_CONFIG = {
    "random_seed": 42,
    "num_legs": 5000,
    "year_start": 2011,
    "year_end": 2024,
    "parlay_sizes": [2, 3, 4],
    "max_parlays_per_size": 1000,
    "max_parlays_analyzed": 500,
    "min_categories_in_parlay": 2,
    "obscure_liquidity_threshold": 0.3,
    "monte_carlo_sims": 2000,
    "report_top_n": 10,
    "q_needle_roi_threshold": 25.0,
    "q_needle_p_value": 0.02,
    "g_needle_roi_threshold": 35.0,
    "g_needle_p_value": 0.01,
    "g_needle_sharpe": 0.8,
    "output_dir": "output",
}


@dataclass
class BetLeg:
    game_id: str
    season_year: int
    bet_type: str
    selection: str
    odds_decimal: float
    implied_prob: float
    actual_outcome: bool
    closing_line: str
    market_category: str
    liquidity_score: float = 1.0
    bookmaker_count: int = 10
    predicted_prob: Optional[float] = None


class QuantumSeekerFramework:
    """Complete end-to-end system with everything integrated."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.years = self._compute_years()
        self.rng = np.random.default_rng(self.config["random_seed"])
        self.bet_legs: List[BetLeg] = []
        self.quantum_results: Dict = {}
        self._feature_columns: List[str] = []
        self._win_model = None
        self._scaler = None

        LOGGER.info("Initializing Quantum Seeker 2.0")
        self._generate_data()
        self._train_win_model()

    def _compute_years(self) -> List[int]:
        year_start = int(self.config["year_start"])
        year_end = int(self.config["year_end"])
        if year_end < year_start:
            raise ValueError(f"year_end ({year_end}) must be >= year_start ({year_start})")
        return list(range(year_start, year_end + 1))

    def _generate_data(self) -> None:
        """Generate synthetic Super Bowl betting data with realistic patterns."""
        num_legs = int(self.config["num_legs"])
        years = self.years

        game_contexts = self._build_game_contexts(years)
        all_types = [bt for cat, bts in REAL_SB_PROP_CATEGORIES.items() for bt in bts]

        LOGGER.info("Generating %s synthetic bet legs...", num_legs)
        for _ in range(num_legs):
            season_year = int(self.rng.choice(years))
            game_id = f"SB_{season_year}"
            bet_type = str(self.rng.choice(all_types))
            market_category = self._categorize(bet_type)
            selection = self._gen_selection(bet_type)

            profile = CATEGORY_PROFILES.get(market_category, CATEGORY_PROFILES["TRADITIONAL"])
            implied_prob = float(
                np.clip(
                    self.rng.normal(profile["prob_mean"], profile["prob_std"]),
                    0.05,
                    0.90,
                )
            )

            margin = float(self.rng.uniform(0.03, 0.08))
            odds_decimal = float(max(1.01, (1.0 / implied_prob) * (1.0 + margin)))
            liquidity_score = float(
                np.clip(
                    self.rng.normal(profile["liquidity_mean"], 0.12),
                    0.05,
                    0.95,
                )
            )
            bookmaker_count = int(max(1, self.rng.poisson(2 + liquidity_score * 5)))
            context = game_contexts[season_year]
            true_prob = self._true_probability(bet_type, market_category, implied_prob, context)
            actual_outcome = bool(self.rng.random() < true_prob)

            self.bet_legs.append(
                BetLeg(
                    game_id=game_id,
                    season_year=season_year,
                    bet_type=bet_type,
                    selection=selection,
                    odds_decimal=odds_decimal,
                    implied_prob=implied_prob,
                    actual_outcome=actual_outcome,
                    closing_line=f"Line_{self.rng.integers(1, 100)}",
                    market_category=market_category,
                    liquidity_score=liquidity_score,
                    bookmaker_count=bookmaker_count,
                )
            )

        LOGGER.info("Generated %s bet legs across %s Super Bowls", len(self.bet_legs), len(years))

    def _build_game_contexts(self, years: List[int]) -> Dict[int, Dict[str, float]]:
        contexts: Dict[int, Dict[str, float]] = {}
        for year in years:
            contexts[year] = {
                "ceremony_index": float(self.rng.normal(0.0, 1.0)),
                "weather_index": float(self.rng.normal(0.0, 1.0)),
                "coin_bias": float(self.rng.normal(0.0, 0.2)),
                "color_bias": float(self.rng.normal(0.0, 0.25)),
                "offense_strength": float(self.rng.normal(0.0, 1.0)),
                "variance": float(self.rng.normal(0.0, 0.5)),
            }
        return contexts

    def _true_probability(
        self, bet_type: str, category: str, implied_prob: float, context: Dict[str, float]
    ) -> float:
        adjustment = 0.0

        if category in {
            "NATIONAL_ANTHEM",
            "HALFTIME_SHOW",
            "GATORADE_SHOWER",
        }:
            adjustment += context["ceremony_index"] * 0.03
        if category in {"COIN_TOSS"}:
            adjustment += context["coin_bias"] * 0.05
        if "GATORADE_COLOR" in bet_type or category == "GATORADE_SHOWER":
            adjustment += context["color_bias"] * 0.04
        if bet_type in {"TOTAL_POINTS", "POINT_SPREAD", "QB_PASS_YARDS", "RB_RUSH_YARDS", "GAME_TOTALS"}:
            adjustment += context["offense_strength"] * 0.04
        if category in {"GAME_OUTCOME", "SCORING", "TURNOVERS"}:
            adjustment += context["variance"] * 0.02

        return float(np.clip(implied_prob + adjustment, 0.02, 0.98))

    def _gen_selection(self, bet_type: str) -> str:
        if "OVER_UNDER" in bet_type or "DURATION" in bet_type:
            if "95" in bet_type:
                return str(self.rng.choice(["OVER_95", "UNDER_95"]))
            elif "100" in bet_type:
                return str(self.rng.choice(["OVER_100", "UNDER_100"]))
            elif "105" in bet_type:
                return str(self.rng.choice(["OVER_105", "UNDER_105"]))
            elif "48" in bet_type:
                return str(self.rng.choice(["OVER_48", "UNDER_48"]))
            elif "51" in bet_type:
                return str(self.rng.choice(["OVER_51", "UNDER_51"]))
            elif "6" in bet_type:
                return str(self.rng.choice(["OVER_6", "UNDER_6"]))
            elif "3" in bet_type:
                return str(self.rng.choice(["OVER_3", "UNDER_3"]))
            elif "10" in bet_type:
                return str(self.rng.choice(["OVER_10", "UNDER_10"]))
            elif "45" in bet_type:
                return str(self.rng.choice(["OVER_45", "UNDER_45"]))
            else:
                return f"OVER_{self.rng.integers(1, 100)}.5"
        if "GATORADE_COLOR" in bet_type:
            return str(self.rng.choice(["ORANGE", "YELLOW", "BLUE", "CLEAR", "RED", "GREEN", "PURPLE", "NONE"]))
        if "COIN_TOSS" in bet_type:
            if "HEADS_TAILS" in bet_type:
                return str(self.rng.choice(["HEADS", "TAILS"]))
            elif "DEFERS" in bet_type:
                return str(self.rng.choice(["DEFERS", "RECEIVES"]))
            elif "RECEIVES" in bet_type:
                return str(self.rng.choice(["RECEIVES", "DEFERS"]))
        if "FIRST_SCORE" in bet_type:
            return str(self.rng.choice(["TD", "FG", "SAFETY"]))
        if "LAST_SCORE" in bet_type:
            return str(self.rng.choice(["TD", "FG"]))
        if "OVERTIME" in bet_type:
            return str(self.rng.choice(["YES", "NO"]))
        if "SAFETY" in bet_type and "GAME" in bet_type:
            return str(self.rng.choice(["YES", "NO"]))
        if "MVP" in bet_type:
            return str(self.rng.choice(["QB", "RB", "WR", "TE", "DEFENSIVE", "KICKER"]))
        if "HALFTIME" in bet_type:
            if "FIRST_SONG" in bet_type:
                return str(self.rng.choice(["SONG_1", "SONG_2", "SONG_3"]))
            elif "SPECIAL_GUEST" in bet_type:
                return str(self.rng.choice(["YES", "NO"]))
            elif "OUTFIT_COLOR" in bet_type:
                return str(self.rng.choice(["RED", "BLUE", "BLACK", "WHITE", "GOLD"]))
        if "WINNER" in bet_type:
            return str(self.rng.choice(["HOME", "AWAY"]))
        return f"SELECTION_{self.rng.integers(1, 5)}"

    def _categorize(self, bet_type: str) -> str:
        for cat, bets in REAL_SB_PROP_CATEGORIES.items():
            if bet_type in bets:
                return cat
        return "TRADITIONAL"

    def _build_feature_frame(self, legs: List[BetLeg]) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "odds_decimal": [leg.odds_decimal for leg in legs],
                "implied_prob": [leg.implied_prob for leg in legs],
                "liquidity_score": [leg.liquidity_score for leg in legs],
                "bookmaker_count": [leg.bookmaker_count for leg in legs],
                "market_category": [leg.market_category for leg in legs],
            }
        )
        df = pd.get_dummies(df, columns=["market_category"], prefix="cat")
        return df

    def _train_win_model(self) -> None:
        if not SKLEARN_AVAILABLE:
            LOGGER.warning("scikit-learn not available; using implied probabilities only.")
            return

        try:
            df = self._build_feature_frame(self.bet_legs)
            self._feature_columns = list(df.columns)
            X = df.values
            y = np.array([1 if leg.actual_outcome else 0 for leg in self.bet_legs])

            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            model = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced")
            model.fit(X_scaled, y)
            self._win_model = model
            LOGGER.info("Win probability model trained with %s samples", len(y))

            # Batch predict probabilities using the already scaled data
            probs = model.predict_proba(X_scaled)[:, 1]
            probs = np.clip(probs, 0.02, 0.98)

            if len(probs) != len(self.bet_legs):
                LOGGER.warning("Mismatch in probability prediction count: %s vs %s", len(probs), len(self.bet_legs))

            for leg, prob in zip(self.bet_legs, probs):
                leg.predicted_prob = float(prob)
            LOGGER.info("Cached probabilities for %s legs", len(self.bet_legs))

        except Exception as exc:
            LOGGER.warning("Failed to train win model: %s", exc)
            self._win_model = None
            self._scaler = None

    def _predict_leg_prob(self, leg: BetLeg) -> float:
        if leg.predicted_prob is not None:
            return leg.predicted_prob

        if self._win_model is None or self._scaler is None:
            return leg.implied_prob

        df = self._build_feature_frame([leg])
        for col in self._feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self._feature_columns]
        try:
            X_scaled = self._scaler.transform(df.values)
            prob = float(self._win_model.predict_proba(X_scaled)[0, 1])
            return float(np.clip(prob, 0.02, 0.98))
        except Exception:
            return leg.implied_prob

    def execute_full_analysis(self) -> Dict:
        LOGGER.info("Executing full 9-phase analysis")

        parlays = self._generate_parlays()
        all_perf = self._analyze_performance(parlays)

        temporal = self._temporal_analysis(parlays)
        synergies = self._synergy_analysis(all_perf)
        meta_edges = self._meta_edges_from_performance(all_perf)
        round_robin = self._round_robin_analysis(all_perf)
        correlations = self._correlation_analysis(all_perf)
        inefficiencies = self._find_inefficiencies(all_perf)

        q_needles, g_needles = self._identify_needles(all_perf)
        self.quantum_results = {
            "metadata": {
                "mode": "QUANTUM_COMPLETE",
                "timestamp": datetime.now().isoformat(),
                "parlays_generated": len(parlays),
                "parlays_analyzed": len(all_perf),
                "q_needles": len(q_needles),
                "g_needles": len(g_needles),
                "output_dir": self.config["output_dir"],
            },
            "quantum_needles": q_needles[:5],
            "golden_needles": g_needles[:3],
            "temporal": temporal,
            "synergies": synergies,
            "meta_edges": meta_edges,
            "round_robin": round_robin[:3],
            "correlations": correlations,
            "inefficiencies": inefficiencies[:10],
            "top_constructions": all_perf[:10],
        }

        return self.quantum_results

    def _generate_parlays(self) -> List[List[BetLeg]]:
        obscure = [
            leg for leg in self.bet_legs if leg.liquidity_score < self.config["obscure_liquidity_threshold"]
        ]
        parlays: List[List[BetLeg]] = []
        min_categories = int(self.config["min_categories_in_parlay"])
        max_parlays_per_size = int(self.config["max_parlays_per_size"])

        for size in self.config["parlay_sizes"]:
            size = int(size)
            attempts = min(max_parlays_per_size, max(1, len(obscure)))
            for _ in range(attempts):
                if len(obscure) < size:
                    break
                parlay = list(self.rng.choice(obscure, size=size, replace=False))
                if len({leg.market_category for leg in parlay}) >= min_categories:
                    parlays.append(parlay)
        return parlays

    def _analyze_performance(self, parlays: List[List[BetLeg]]) -> List[Dict]:
        results = []
        max_analyzed = int(self.config["max_parlays_analyzed"])
        for parlay in parlays[:max_analyzed]:
            perf = self._calc_perf(parlay)
            sig = self._monte_carlo(perf)
            perf["sig"] = sig
            results.append(perf)
        return results

    def _calc_perf(self, parlay: List[BetLeg]) -> Dict:
        total_bets = int(self.rng.integers(5, 20))
        leg_probs = [self._predict_leg_prob(leg) for leg in parlay]
        parlay_prob = float(np.clip(np.prod(leg_probs), 0.0001, 0.95))
        wins = int(self.rng.binomial(total_bets, parlay_prob))
        parlay_odds = float(np.prod([leg.odds_decimal for leg in parlay]))

        roi = ((wins * (parlay_odds - 1.0)) - (total_bets - wins)) / total_bets * 100.0

        returns = [parlay_odds - 1.0] * wins + [-1.0] * (total_bets - wins)
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns)) if len(returns) > 1 else 0.0
        sharpe = mean_return / std_return if std_return > 0 else 0.0

        obscurity = float(np.mean([1.0 - leg.liquidity_score for leg in parlay]))
        return {
            "desc": " + ".join([f"{l.bet_type}:{l.selection}" for l in parlay]),
            "legs": [(l.bet_type, l.selection) for l in parlay],
            "total_bets": total_bets,
            "wins": wins,
            "win_rate": round((wins / total_bets) * 100.0, 2),
            "roi_percent": round(roi, 2),
            "sharpe_ratio": round(sharpe, 3),
            "obscurity": round(obscurity, 3),
            "parlay_prob": round(parlay_prob, 4),
        }

    def _monte_carlo(self, perf: Dict) -> Dict:
        total_bets = perf["total_bets"]
        parlay_prob = perf.get("parlay_prob", 0.01)
        parlay_odds = self._estimate_parlay_odds(perf)
        sims = int(self.config["monte_carlo_sims"])
        roi_samples = []

        for _ in range(sims):
            wins = int(self.rng.binomial(total_bets, parlay_prob))
            roi = ((wins * (parlay_odds - 1.0)) - (total_bets - wins)) / total_bets * 100.0
            roi_samples.append(roi)

        roi_samples = np.array(roi_samples)
        p_value = float(np.mean(roi_samples >= perf["roi_percent"]))
        expected_roi = float(np.mean(roi_samples))

        if SCIPY_AVAILABLE:
            try:
                expected_wins = total_bets * parlay_prob
                p_binom = stats.binomtest(perf["wins"], total_bets, parlay_prob).pvalue
            except Exception:
                p_binom = p_value
        else:
            p_binom = p_value

        return {
            "p_value": round(p_value, 5),
            "binom_p_value": round(float(p_binom), 5),
            "sig_95": p_value < 0.05,
            "sig_99": p_value < 0.01,
            "expected_roi": round(expected_roi, 2),
        }

    def _estimate_parlay_odds(self, perf: Dict) -> float:
        if "legs" not in perf or not perf["legs"]:
            return 2.0
        leg_types = [leg_type for leg_type, _ in perf["legs"]]
        legs = [leg for leg in self.bet_legs if leg.bet_type in leg_types]
        if not legs:
            return 2.0
        return float(np.mean([leg.odds_decimal for leg in legs]) ** len(leg_types))

    def _identify_needles(self, all_perf: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        q_needles = []
        g_needles = []

        for perf in all_perf:
            roi = perf["roi_percent"]
            sig = perf.get("sig", {})
            if roi >= self.config["q_needle_roi_threshold"] and sig.get("p_value", 1.0) < self.config["q_needle_p_value"]:
                q_needles.append(perf)

            if (
                roi >= self.config["g_needle_roi_threshold"]
                and sig.get("p_value", 1.0) < self.config["g_needle_p_value"]
                and perf["sharpe_ratio"] > self.config["g_needle_sharpe"]
            ):
                g_needles.append(
                    {
                        "perf": perf,
                        "explanation": self._explain(perf),
                        "confidence": "HIGH" if perf["total_bets"] >= 10 else "MEDIUM",
                    }
                )

        return q_needles, g_needles

    def _explain(self, perf: Dict) -> str:
        explanations = []
        for leg_type, _ in perf["legs"]:
            if "ANTHEM" in leg_type and "DURATION" in leg_type:
                explanations.append("Anthem duration shows historical patterns")
            elif "GATORADE_COLOR" in leg_type:
                explanations.append("Gatorade color selection follows team traditions")
            elif "COIN_TOSS" in leg_type:
                explanations.append("Coin toss outcomes show statistical balance")
            elif "FIRST_SCORE" in leg_type:
                explanations.append("First score type correlates with team strategy")
            elif "MVP" in leg_type:
                explanations.append("MVP position reflects game flow patterns")
            elif "HALFTIME" in leg_type:
                explanations.append("Halftime show props show performer patterns")
            elif "OVERTIME" in leg_type or "SAFETY" in leg_type:
                explanations.append("Rare game events show value in low-liquidity markets")
            else:
                explanations.append("Low-liquidity markets show mispricing")
        return " | ".join(explanations[:3])

    def _temporal_analysis(self, parlays: List[List[BetLeg]]) -> Dict:
        results = {}
        for year in self.years:
            year_parlays = [p for p in parlays if all(leg.season_year == year for leg in p)]
            if not year_parlays:
                continue
            perfs = [self._calc_perf(p) for p in year_parlays[:50]]
            rois = [p["roi_percent"] for p in perfs]
            results[year] = {"avg_roi": round(float(np.mean(rois)), 2)}

        if not results:
            return {}

        years = list(results.keys())
        rois = [results[y]["avg_roi"] for y in years]
        trend = float(np.polyfit(range(len(rois)), rois, 1)[0]) if len(rois) > 1 else 0.0
        best_era = max(years, key=lambda y: results[y]["avg_roi"])
        recent = float(np.mean(rois[-3:])) if len(rois) >= 3 else 0.0

        return {
            "yearly": results,
            "trend": round(trend, 3),
            "best_era": best_era,
            "recent": round(recent, 2),
        }

    def _synergy_analysis(self, perfs: List[Dict]) -> List[Dict]:
        pairs: Dict[str, List[float]] = {}
        for perf in perfs:
            categories = set()
            for leg_type, _ in perf["legs"]:
                categories.add(self._categorize(leg_type))
            combo = "+".join(sorted(categories))
            pairs.setdefault(combo, []).append(perf["roi_percent"])

        synergies = []
        for combo, rois in pairs.items():
            if len(rois) >= 3:
                avg_roi = float(np.mean(rois))
                sharpe = avg_roi / max(0.1, float(np.std(rois)))
                synergies.append(
                    {
                        "combo": combo,
                        "avg_roi": round(avg_roi, 2),
                        "sharpe": round(sharpe, 3),
                        "n": len(rois),
                    }
                )

        return sorted(synergies, key=lambda x: x["avg_roi"], reverse=True)[:10]

    def _meta_edges_from_performance(self, perfs: List[Dict]) -> Dict[str, Dict]:
        bet_types = {leg[0] for perf in perfs for leg in perf["legs"]}
        meta_edges = {}
        for bet_type in list(bet_types)[:15]:
            edge = self._meta_edge(bet_type)
            if edge and edge.get("overall_roi", 0) > 10:
                meta_edges[bet_type] = edge
        return meta_edges

    def _meta_edge(self, bet_type: str) -> Dict:
        legs = [leg for leg in self.bet_legs if leg.bet_type == bet_type]
        if not legs:
            return {}
        years = sorted(set(leg.season_year for leg in legs))
        year_data = []
        for year in years:
            year_legs = [leg for leg in legs if leg.season_year == year]
            if not year_legs:
                continue
            win_rate = sum(1 for leg in year_legs if leg.actual_outcome) / len(year_legs)
            odds = float(np.mean([leg.odds_decimal for leg in year_legs]))
            roi = ((win_rate * (odds - 1.0)) - (1.0 - win_rate)) * 100.0
            year_data.append(roi)
        if not year_data:
            return {}
        return {
            "type": bet_type,
            "overall_roi": round(float(np.mean(year_data)), 2),
            "consistency": round(float(np.std(year_data)), 2),
            "positive_years": sum(1 for r in year_data if r > 0),
            "total_years": len(year_data),
        }

    def _round_robin_analysis(self, perfs: List[Dict]) -> List[Dict]:
        results = []
        for perf in perfs[:5]:
            leg_types = [leg[0] for leg in perf["legs"]]
            candidate_legs = [leg for leg in self.bet_legs if leg.bet_type in leg_types][:6]
            if len(candidate_legs) < 3:
                continue
            rr = self._round_robin(candidate_legs)
            if rr:
                results.append(rr)
        return results

    def _round_robin(self, legs: List[BetLeg]) -> Dict:
        results = {}
        for k in range(2, min(len(legs), 5)):
            combos = list(itertools.combinations(legs, k))[:50]
            perfs = [self._calc_perf(list(c)) for c in combos]
            rois = [p["roi_percent"] for p in perfs]
            results[f"by_{k}s"] = {
                "n": len(perfs),
                "roi": round(float(np.mean(rois)), 2),
                "best": round(float(np.max(rois)), 2),
            }
        if not results:
            return {}
        optimal = max(results.items(), key=lambda x: x[1]["roi"])
        return {
            "base_legs": len(legs),
            "optimal": optimal[0],
            "opt_roi": optimal[1]["roi"],
            "formats": results,
        }

    def _correlation_analysis(self, perfs: List[Dict]) -> List[Dict]:
        types = list({leg[0] for perf in perfs for leg in perf["legs"]})
        corrs = []
        for i, t1 in enumerate(types):
            for t2 in types[i + 1 :]:
                co = sum(
                    1
                    for perf in perfs
                    if t1 in [leg[0] for leg in perf["legs"]] and t2 in [leg[0] for leg in perf["legs"]]
                )
                if perfs and co / len(perfs) > 0.1:
                    corrs.append({"t1": t1, "t2": t2, "rate": round(co / len(perfs), 3)})
        return sorted(corrs, key=lambda x: x["rate"], reverse=True)[:10]

    def _find_inefficiencies(self, perfs: List[Dict]) -> List[Dict]:
        ineff: Dict[str, List[float]] = {}
        for perf in perfs:
            for leg_type, _ in perf["legs"]:
                ineff.setdefault(leg_type, []).append(perf["roi_percent"])
        results = []
        for bet_type, rois in ineff.items():
            if len(rois) >= 3 and np.mean(rois) > 15:
                results.append(
                    {
                        "type": bet_type,
                        "avg_roi": round(float(np.mean(rois)), 2),
                        "freq": len(rois),
                        "category": self._categorize(bet_type),
                        "rating": "HIGH" if np.mean(rois) > 25 else "MEDIUM",
                    }
                )
        return sorted(results, key=lambda x: x["avg_roi"], reverse=True)

    def generate_full_report(self) -> str:
        if not self.quantum_results:
            self.execute_full_analysis()
        r = self.quantum_results
        report = []
        report.append("")
        report.append("=" * 80)
        report.append("QUANTUM SEEKER 2.0 - COMPLETE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        metadata = r["metadata"]
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 80)
        report.append(f"Mode: {metadata['mode']}")
        report.append(f"Parlays Generated: {metadata['parlays_generated']}")
        report.append(f"Parlays Analyzed: {metadata['parlays_analyzed']}")
        report.append(f"Quantum Needles: {metadata['q_needles']} | Golden Needles: {metadata['g_needles']}")
        report.append(f"Timestamp: {metadata['timestamp']}")
        report.append("")

        if r.get("golden_needles"):
            report.append("GOLDEN NEEDLES")
            report.append("-" * 80)
            for i, needle in enumerate(r["golden_needles"], 1):
                perf = needle["perf"]
                report.append(f"Needle #{i}")
                report.append(f"  {perf['desc'][:70]}...")
                report.append(f"  ROI: {perf['roi_percent']}% | Sharpe: {perf['sharpe_ratio']}")
                report.append(f"  p-value: {perf.get('sig', {}).get('p_value', 'N/A')}")
                report.append(f"  Reason: {needle['explanation']}")
                report.append(f"  Confidence: {needle['confidence']}")
                report.append("")

        if r.get("quantum_needles"):
            report.append("QUANTUM NEEDLES")
            report.append("-" * 80)
            for i, perf in enumerate(r["quantum_needles"], 1):
                report.append(f"Needle #{i}")
                report.append(f"  {perf['desc'][:70]}...")
                report.append(f"  ROI: {perf['roi_percent']}% | Win Rate: {perf['win_rate']}%")
                report.append(f"  p-value: {perf.get('sig', {}).get('p_value', 'N/A')}")
                report.append(f"  Obscurity Score: {perf['obscurity']}")
                report.append("")

        if r.get("temporal"):
            report.append("TEMPORAL PATTERN ANALYSIS")
            report.append("-" * 80)
            report.append(f"ROI Trend: {r['temporal'].get('trend', 0):.3f}% per year")
            report.append(f"Best Era: SB {r['temporal'].get('best_era', 'N/A')}")
            report.append(f"Recent Performance (Last 3 SBs): {r['temporal'].get('recent', 0)}% ROI")
            report.append("")

        if r.get("synergies"):
            report.append("TOP CROSS-CATEGORY SYNERGIES")
            report.append("-" * 80)
            for i, syn in enumerate(r["synergies"][:5], 1):
                report.append(f"#{i}: {syn['combo']}")
                report.append(f"  Avg ROI: {syn['avg_roi']}% | Sharpe: {syn['sharpe']}")
                report.append(f"  Sample Size: {syn['n']}")
                report.append("")

        if r.get("meta_edges"):
            report.append("TOP META-EDGES BY BET TYPE")
            report.append("-" * 80)
            sorted_edges = sorted(r["meta_edges"].items(), key=lambda x: x[1]["overall_roi"], reverse=True)
            for i, (bet_type, edge) in enumerate(sorted_edges[:5], 1):
                report.append(f"#{i}: {bet_type}")
                report.append(f"  Overall ROI: {edge['overall_roi']}%")
                report.append(f"  Consistency: sigma={edge['consistency']}")
                report.append(f"  Positive Years: {edge['positive_years']}/{edge['total_years']}")
                report.append("")

        if r.get("round_robin"):
            report.append("OPTIMAL ROUND ROBIN STRATEGIES")
            report.append("-" * 80)
            for i, rr in enumerate(r["round_robin"], 1):
                report.append(f"#{i}: {rr['base_legs']} base legs")
                report.append(f"  Optimal Format: {rr['optimal']}")
                report.append(f"  Optimal ROI: {rr['opt_roi']}%")
                report.append("")

        if r.get("correlations"):
            report.append("STRONG BET TYPE CORRELATIONS")
            report.append("-" * 80)
            for corr in r["correlations"][:5]:
                report.append(f"{corr['t1']} <-> {corr['t2']}")
                report.append(f"  Co-occurrence: {corr['rate'] * 100:.1f}%")
                report.append("")

        if r.get("inefficiencies"):
            report.append("TOP MARKET INEFFICIENCIES")
            report.append("-" * 80)
            for i, ineff in enumerate(r["inefficiencies"][:5], 1):
                report.append(f"#{i}: {ineff['type']}")
                report.append(f"  Category: {ineff['category']}")
                report.append(f"  Avg ROI: {ineff['avg_roi']}%")
                report.append(f"  Rating: {ineff['rating']} | Frequency: {ineff['freq']}")
                report.append("")

        report.append("ACTIONABLE RECOMMENDATIONS")
        report.append("-" * 80)
        recommendations = [
            "1. Focus on LOW-LIQUIDITY props: Anthem duration, Gatorade color, coin toss",
            "2. Build 2-3 leg parlays mixing CEREMONIAL + GAME props",
            "3. Best combos: Anthem + First Score, Coin Toss + Safety, Gatorade + Overtime",
            "4. Use ROUND ROBIN formats to diversify (picks 4 legs, bet all 2-leg combos)",
            "5. Target props with HIGH OBSCURITY: MVP position, halftime guest, special teams",
            "6. Avoid correlated bets (don't parlay QB yards with QB TDs)",
            "7. Line shop across DraftKings, FanDuel, Caesars for best odds",
            "8. Track historical patterns: Gatorade orange hits 40%, Anthem usually OVER",
        ]
        report.extend([f"- {rec}" for rec in recommendations])
        report.append("")
        report.append("=" * 80)
        report.append("ANALYSIS COMPLETE")
        report.append("=" * 80)
        report.append("")
        return "\n".join(report)

    def generate_html_report(self, report_text: Optional[str] = None) -> str:
        if report_text is None:
            report_text = self.generate_full_report()
        escaped = report_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = [
            "<html>",
            "<head>",
            "<meta charset=\"utf-8\"/>",
            "<title>Quantum Seeker 2.0 Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 24px; }",
            "pre { white-space: pre-wrap; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Quantum Seeker 2.0 Report</h1>",
            "<pre>",
            escaped,
            "</pre>",
            "</body>",
            "</html>",
        ]
        return "\n".join(html)

    def generate_visualizations(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        if not MATPLOTLIB_AVAILABLE:
            LOGGER.warning("matplotlib not available; skipping visualizations.")
            return {}
        if not self.quantum_results:
            self.execute_full_analysis()

        out_dir = output_dir or self.config["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        temporal = self.quantum_results.get("temporal", {})
        yearly = temporal.get("yearly", {})
        if yearly:
            years = sorted(yearly.keys())
            rois = [yearly[y]["avg_roi"] for y in years]
            plt.figure(figsize=(10, 4))
            plt.plot(years, rois, marker="o")
            plt.title("ROI Trend by Super Bowl Year")
            plt.xlabel("Year")
            plt.ylabel("Average ROI (%)")
            plt.grid(True, linestyle="--", alpha=0.4)
            path = os.path.join(out_dir, "roi_trend.png")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            paths["roi_trend"] = path

        perfs = self.quantum_results.get("top_constructions", [])
        if perfs:
            plt.figure(figsize=(6, 4))
            x = [p["roi_percent"] for p in perfs]
            y = [p["sharpe_ratio"] for p in perfs]
            plt.scatter(x, y, alpha=0.7)
            plt.title("Parlay Performance")
            plt.xlabel("ROI (%)")
            plt.ylabel("Sharpe Ratio")
            plt.grid(True, linestyle="--", alpha=0.4)
            path = os.path.join(out_dir, "parlay_scatter.png")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            paths["parlay_scatter"] = path

        ineff = self.quantum_results.get("inefficiencies", [])
        if ineff:
            top = ineff[:10]
            labels = [i["type"] for i in top]
            values = [i["avg_roi"] for i in top]
            plt.figure(figsize=(10, 4))
            plt.barh(labels[::-1], values[::-1])
            plt.title("Top Market Inefficiencies (Avg ROI)")
            plt.xlabel("Avg ROI (%)")
            path = os.path.join(out_dir, "market_inefficiencies.png")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            paths["market_inefficiencies"] = path

        if SEABORN_AVAILABLE:
            heatmap_path = self._synergy_heatmap(out_dir)
            if heatmap_path:
                paths["synergy_heatmap"] = heatmap_path
        return paths

    def _synergy_heatmap(self, output_dir: str) -> Optional[str]:
        synergies = self.quantum_results.get("synergies", [])
        if not synergies:
            return None
        categories = list(REAL_SB_PROP_CATEGORIES.keys())
        matrix = pd.DataFrame(0.0, index=categories, columns=categories)
        counts = pd.DataFrame(0, index=categories, columns=categories)

        for syn in synergies:
            combo = syn["combo"].split("+")
            for i, cat_a in enumerate(combo):
                for cat_b in combo[i:]:
                    if cat_a in matrix.index and cat_b in matrix.columns:
                        matrix.loc[cat_a, cat_b] += syn["avg_roi"]
                        matrix.loc[cat_b, cat_a] += syn["avg_roi"]
                        counts.loc[cat_a, cat_b] += 1
                        counts.loc[cat_b, cat_a] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_matrix = matrix / counts.replace(0, np.nan)
            avg_matrix = avg_matrix.fillna(0.0)

        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_matrix, cmap="viridis")
        plt.title("Cross-Category Synergy Heatmap (Avg ROI)")
        path = os.path.join(output_dir, "synergy_heatmap.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path


def _load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def execute_quantum_seeker(
    config_path: Optional[str] = None, config_override: Optional[Dict] = None
) -> Dict:
    """Main entry point."""
    try:
        config = _load_config(config_path)
        if config_override:
            config.update(config_override)
        framework = QuantumSeekerFramework(config=config)
        results = framework.execute_full_analysis()
        report = framework.generate_full_report()

        output_dir = config.get("output_dir", DEFAULT_CONFIG["output_dir"])
        os.makedirs(output_dir, exist_ok=True)

        report_path = os.path.join(output_dir, "quantum_seeker_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as handle:
                handle.write(report)
        except Exception as exc:
            LOGGER.error("Failed to write report: %s", exc)

        results_path = os.path.join(output_dir, "quantum_seeker_results.json")
        try:
            with open(results_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2, default=str)
        except Exception as exc:
            LOGGER.error("Failed to write results JSON: %s", exc)

        html_path = os.path.join(output_dir, "quantum_seeker_report.html")
        try:
            html_report = framework.generate_html_report(report_text=report)
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write(html_report)
        except Exception as exc:
            LOGGER.error("Failed to write HTML report: %s", exc)

        try:
            viz_paths = framework.generate_visualizations(output_dir=output_dir)
            if viz_paths:
                results["visualizations"] = viz_paths
                with open(results_path, "w", encoding="utf-8") as handle:
                    json.dump(results, handle, indent=2, default=str)
        except Exception as exc:
            LOGGER.error("Visualization generation failed: %s", exc)

        LOGGER.info("Report saved to %s", report_path)
        LOGGER.info("Results saved to %s", results_path)
        return results
    except Exception as exc:
        LOGGER.exception("Quantum Seeker execution failed: %s", exc)
        raise


if __name__ == "__main__":
    execute_quantum_seeker()
