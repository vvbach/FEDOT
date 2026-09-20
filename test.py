# test.py

import numpy as np

from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.api.main import Fedot


def generate_dataset(
    n_samples: int = 300,
    random_seed: int = 42,
):
    rng = np.random.default_rng(random_seed)

    feature_1 = rng.normal(0, 1, n_samples)
    feature_2 = rng.uniform(-5, 5, n_samples)
    feature_3 = rng.exponential(2, n_samples)
    feature_4 = rng.integers(0, 10, n_samples)
    feature_5 = rng.normal(10, 3, n_samples)
    feature_6 = rng.uniform(0, 1, n_samples)
    feature_7 = rng.normal(0, 5, n_samples)
    feature_8 = rng.uniform(-10, 10, n_samples)

    features = np.column_stack([
        feature_1,
        feature_2,
        feature_3,
        feature_4,
        feature_5,
        feature_6,
        feature_7,
        feature_8,
    ])

    scores = (
        2.0 * feature_1
        + 1.5 * feature_2
        - 0.8 * feature_3
        + 0.5 * feature_4
        + 0.3 * (feature_5 - 10)
        + 2.0 * feature_6
        + 0.2 * feature_1 ** 2
        - 0.5 * feature_2 * feature_6
        + rng.normal(0, 1.5, n_samples)
    )

    target = (scores > np.median(scores)).astype(int)

    return features, target


def generate_regression_dataset(
    n_samples: int = 300,
    random_seed: int = 7,
):
    rng = np.random.default_rng(random_seed)
    features = rng.uniform(-3, 3, size=(n_samples, 6))
    target = (
        3.0 * features[:, 0] ** 2
        - 1.5 * features[:, 1]
        + np.sin(features[:, 2] * 2.0)
        + features[:, 3] * features[:, 4]
        + rng.normal(0, 0.4, n_samples)
    )
    return features, target


def train_test_split(features, target, test_size=0.2, random_seed=42):

    rng = np.random.default_rng(random_seed)

    indices = np.arange(len(target))
    rng.shuffle(indices)

    split_index = int(len(indices) * (1 - test_size))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    return (
        features[train_indices],
        features[test_indices],
        target[train_indices],
        target[test_indices],
    )


def print_recent_generations(history, limit=5):
    generations = history.generations[-limit:]
    if not generations:
        print("No generations recorded.")
        return

    def fitness_value(individual):
        values = individual.fitness.values
        value = values[0] if isinstance(values, (tuple, list)) else values
        return float(value) if value is not None else float("inf")

    print(f"\nLast {len(generations)} generations comparison (lower fitness is better):")
    print("generation | label | best | average | worst | best parameters")
    print("-" * 110)

    for generation in generations:
        individuals = list(generation)
        scores = [fitness_value(individual) for individual in individuals]
        best_individual = min(individuals, key=fitness_value)
        parameters = [
            f"{node.name}: {node.parameters}"
            for node in best_individual.graph.nodes
            if node.parameters
        ]
        parameter_text = "; ".join(parameters) or "{}"
        print(
            f"{generation.generation_num:10} | {str(generation.label):20} | "
            f"{min(scores):.6f} | {np.mean(scores):.6f} | {max(scores):.6f} | "
            f"{parameter_text}"
        )


def run_regression_pipeline():
    """Chay pipeline FEDOT hoi quy voi bo du lieu khac."""
    features, target = generate_regression_dataset()
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_seed=7
    )
    task = Task(TaskTypesEnum.regression)
    train_data = InputData.from_numpy(train_features, train_target, task=task)
    test_data = InputData.from_numpy(test_features, test_target, task=task)

    print("\n" + "=" * 70)
    print("FEDOT REGRESSION PIPELINE TEST")
    print("=" * 70)
    model = Fedot(
        problem="regression",
        timeout=2,
        seed=7,
        logging_level=20,
        with_tuning=True,
    )
    pipeline = model.fit(features=train_data, target=train_target)
    prediction = model.predict(features=test_data)
    rmse = np.sqrt(np.mean((prediction - test_target) ** 2))
    print("Best regression pipeline:")
    print(pipeline)
    print(f"Regression RMSE: {rmse:.4f}")
    print_recent_generations(model.history)


def main():
    features, target = generate_dataset(
        n_samples=300,
        random_seed=42,
    )

    (
        train_features,
        test_features,
        train_target,
        test_target,
    ) = train_test_split(
        features,
        target,
        test_size=0.2,
        random_seed=42,
    )

    task = Task(TaskTypesEnum.classification)

    train_data = InputData.from_numpy(
        train_features,
        train_target,
        task=task,
    )

    test_data = InputData.from_numpy(
        test_features,
        test_target,
        task=task,
    )

    print("=" * 70)
    print("FEDOT FULL PIPELINE TEST")
    print("=" * 70)
    print(f"Train features: {train_features.shape}")
    print(f"Test features:  {test_features.shape}")
    print(f"Train targets:  {np.unique(train_target)}")
    print(f"Test targets:   {np.unique(test_target)}")

    model = Fedot(
        problem="classification",
        timeout=2,
        seed=42,
        logging_level=20,
        with_tuning=True,
    )

    print("\nStarting FEDOT pipeline composition...")

    pipeline = model.fit(
       features=train_data,
       target=train_target,
    )

    print("\nPipeline composition completed.")
    print("Best pipeline:")
    print(pipeline)


    print("\nMaking predictions...")

    prediction = model.predict(
       features=test_data,
    )

    print("Predictions:")
    print(prediction)


    accuracy = np.mean(
        prediction == test_target
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.4f}")


    print("\nChecking optimization history...")

    try:
        history = model.history

        print("Number of generations:", len(history.generations))

        for generation in history.generations:
            print(
                generation.generation_num,
                generation.label,
                len(generation),
                generation.metadata,
            )

        print("Tuning start:", history.tuning_start)
        print("Tuning result:", history.tuning_result)
        print_recent_generations(history)
    except AttributeError:
        print("History is not directly available through model.history.")

    run_regression_pipeline()


if __name__ == "__main__":
    main()