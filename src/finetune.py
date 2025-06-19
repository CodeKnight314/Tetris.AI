import optuna
from src.env import TetrisEnv
from typing import List
import yaml
import os

def create_trial_config(trial_number: int, weights: List[float], base_config_path: str, base_output_dir: str):
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    config["reward_weights"] = weights
    trial_dir = os.path.join(base_output_dir, f"trial_{trial_number:05d}")
    os.makedirs(trial_dir, exist_ok=True)
    
    trial_config_path = os.path.join(trial_dir, f"trial_{trial_number:05d}.yaml")
    with open(trial_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return trial_config_path, trial_dir
    
def objective(trial):
    weights = [
        trial.suggest_float('lines_cleared', 1, 10),
        trial.suggest_float('holes', -10, -1),
        trial.suggest_float('bumpiness', -10, -1),
        trial.suggest_float('aggregate_height', -10, -1),
        trial.suggest_float('max_height', -10, -1),
        trial.suggest_float('tetris_bonus', 4, 10),
        trial.suggest_float('survival_bonus', 0, 1),
        trial.suggest_float('line_progress', 0, 5),
        trial.suggest_float('low_placement', 0, 5)
    ]
    
    base_config = "src/config/finetune.yaml"
    base_output_dir = "../optuna_trials"
    
    try:
        trial_config_path, trial_dir = create_trial_config(
            trial.number, weights, base_config, base_output_dir
        )
        
        env = TetrisEnv(seed=1898, num_envs=128, config=trial_config_path, verbose=False)
        lines_cleared = env.train(path=trial_dir)
        
        results = {
            'trial_number': trial.number,
            'weights': weights,
            'lines_cleared': lines_cleared,
            'config_path': trial_config_path
        }
        
        results_path = os.path.join(trial_dir, "results.yaml")
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        return lines_cleared
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        print(f"  Error type: {type(e).__name__}")
        try:
            failure_info = {
                'trial_number': trial.number,
                'weights': weights,
                'error': str(e),
                'error_type': type(e).__name__,
                'lines_cleared': 0
            }
            failure_path = os.path.join(base_output_dir, f"trial_{trial.number:03d}", "failure.yaml")
            os.makedirs(os.path.dirname(failure_path), exist_ok=True)
            with open(failure_path, 'w') as f:
                yaml.dump(failure_info, f, default_flow_style=False)
        except:
            pass 
        
        return 0
    
    finally:
        env.close()
    
def main():
    output_dir = "../optuna_trials"
    os.makedirs(output_dir, exist_ok=True)
    
    base_config = "src/config/finetune.yaml"
    if not os.path.exists(base_config):
        print(f"ERROR: Base config not found: {base_config}")
        print("Please make sure the config file exists before running optimization.")
        return
    
    study = optuna.create_study(
        direction='maximize',
        study_name='tetris_weight_optimization',
        storage=f'sqlite:///{output_dir}/optuna_study.db',
        load_if_exists=True
    )
    
    print("Starting Optuna optimization...")
    print(f"Base config: {base_config}")
    print(f"Results will be saved to: {output_dir}")
    print(f"Number of existing trials: {len(study.trials)}")
    
    try:
        study.optimize(objective, n_trials=75)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nOptimization failed with error: {e}")
        return
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    
    if study.best_trial is None:
        print("No successful trials completed.")
        return
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value} lines cleared")
    print(f"Best weights: {study.best_trial.params}")
    
    best_weights = [
        study.best_trial.params['lines_cleared'],
        study.best_trial.params['holes'],
        study.best_trial.params['bumpiness'],
        study.best_trial.params['aggregate_height'],
        study.best_trial.params['max_height'],
        study.best_trial.params['tetris_bonus'],
        study.best_trial.params['survival_bonus'],
        study.best_trial.params['line_progress'],
        study.best_trial.params['low_placement']
    ]
    
    final_results = {
        'best_trial_number': study.best_trial.number,
        'best_lines_cleared': study.best_value,
        'best_weights': best_weights,
        'best_params': study.best_trial.params,
        'total_trials': len(study.trials),
        'successful_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    }
    
    with open(f"{output_dir}/final_results.yaml", 'w') as f:
        yaml.dump(final_results, f, default_flow_style=False)
    
    print(f"\nFinal results saved to: {output_dir}/final_results.yaml")
    print(f"Best config available at: {output_dir}/trial_{study.best_trial.number:03d}/")

if __name__ == "__main__":
    main()