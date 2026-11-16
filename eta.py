"""Learn delivery ETA from a reproducible synthetic trip dataset."""
import argparse
import json
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def sample_trips(count=1600, seed=42):
    rng = random.Random(seed)
    inputs, minutes = [], []
    for _ in range(count):
        distance = round(rng.uniform(1, 25), 2)
        traffic = rng.randrange(3)
        stops = rng.randrange(9)
        rain = rng.randrange(2)
        duration = (
            6 + distance * (1.7 + 0.8 * traffic + 0.35 * rain)
            + 2.1 * stops + 2 * rain + rng.gauss(0, 2)
        )
        inputs.append([distance, traffic, stops, rain])
        minutes.append(round(duration, 2))
    return inputs, minutes


def train():
    inputs, targets = sample_trips()
    train_x, test_x, train_y, test_y = train_test_split(
        inputs, targets, test_size=0.25, random_state=42
    )
    forest = RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=42)
    forest.fit(train_x, train_y)
    return forest, {
        "holdout_trips": len(test_y),
        "forest_mae_minutes": round(mean_absolute_error(test_y, forest.predict(test_x)), 2),
        "distance_baseline_mae_minutes": round(
            mean_absolute_error(test_y, [distance * 2.5 + 14 for distance, *_ in test_x]), 2
        ),
    }


def estimate(model, distance, traffic, stops, rain):
    if not 1 <= distance <= 25 or traffic not in (0, 1, 2) or stops not in range(9) or rain not in (0, 1):
        raise ValueError("Use 1–25 km, traffic 0–2, stops 0–8, and rain 0 or 1.")
    return round(float(model.predict([[distance, traffic, stops, rain]])[0]), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("evaluate", help="Compare model and baseline on a holdout set")
    predict = subcommands.add_parser("predict", help="Predict minutes for a simulated trip")
    predict.add_argument("--distance", type=float, required=True, help="Distance in km")
    predict.add_argument("--traffic", choices=["light", "moderate", "heavy"], required=True)
    predict.add_argument("--stops", type=int, default=0)
    predict.add_argument("--rain", action="store_true")
    args = parser.parse_args()
    model, metrics = train()
    if args.command == "evaluate":
        print(json.dumps(metrics, indent=2))
        return
    traffic = ("light", "moderate", "heavy").index(args.traffic)
    try:
        minutes = estimate(model, args.distance, traffic, args.stops, int(args.rain))
    except ValueError as error:
        parser.error(str(error))
    print(f"Estimated simulated delivery time: {minutes:.1f} minutes")


if __name__ == "__main__":
    main()
