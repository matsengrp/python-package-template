# Example script to run the experiment
if __name__ == "__main__":
    from mean_estimator.experiment import run_experiment
    
    # Run experiment with different learning rates
    results = run_experiment(
        learning_rates=[0.01, 0.05, 0.1, 0.5],
        epochs=200,
        initial_guess=0.0,
        true_mean=1.0
    )
