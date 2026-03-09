"""
Smoke tests for the state tracking codebase.

- Verifies that conda env definition files (base.yml, fla.yml) exist and define
  the expected env names so anyone cloning the repo can create base and fla2.
- Runs a minimal training step for the tree model when CUDA is available,
  to confirm the pipeline works.

Run from repository root:
  pytest state_tracking/tests/test_smoke.py -v
  # or, without pytest:
  python state_tracking/tests/test_smoke.py
"""
import subprocess
import sys
from pathlib import Path


def _repo_root():
    """Repository root (parent of state_tracking/)."""
    return Path(__file__).resolve().parent.parent.parent


def test_env_files_exist_and_define_base_and_fla2():
    """Conda envs base and fla2 can be created from repo YAML files."""
    root = _repo_root()
    base_yml = root / "base.yml"
    fla_yml = root / "fla.yml"
    assert base_yml.is_file(), f"Expected {base_yml} to exist so 'base' env can be created (conda env create -f base.yml)"
    assert fla_yml.is_file(), f"Expected {fla_yml} to exist so 'fla2' env can be created (conda env create -f fla.yml)"
    base_content = base_yml.read_text()
    fla_content = fla_yml.read_text()
    assert "name: base" in base_content or "name: base\n" in base_content.split("\n")[0], \
        "base.yml should define env name 'base'"
    assert "name: fla2" in fla_content or "name: fla2\n" in fla_content.split("\n")[0], \
        "fla.yml should define env name 'fla2'"


def test_training_smoke_tree():
    """One training step with tree model to verify the training pipeline runs (requires CUDA)."""
    import torch
    if not torch.cuda.is_available():
        try:
            import pytest
            pytest.skip("Training smoke test requires CUDA; run on a machine with GPU to validate training pipeline")
        except ImportError:
            return  # no pytest: skip by returning
    root = _repo_root()
    out_dir = root / "state_tracking" / "saved_models" / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "state_tracking.train",
                "--model", "tree",
                "--num_items", "5",
                "--max_len", "2",
                "--chunk_size", "1",
                "--T1_num_layers", "1",
                "--T2_num_layers", "1",
                "--num_stories", "100",
                "--epochs", "1",
                "--max_steps", "1",
                "--batch_size", "4",
                "--generate_dataset",
                "--output_dir", str(out_dir),
                "--dataset_root", str(root / "state_tracking" / "datasets"),
                "--disable_wandb",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"Training smoke test failed (exit {result.returncode}). "
            f"stdout: {result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout} "
            f"stderr: {result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr}"
        )
    finally:
        # Leave saved_models/smoke_test in place; it's gitignored and small
        pass


if __name__ == "__main__":
    # Run without pytest
    test_env_files_exist_and_define_base_and_fla2()
    print("test_env_files_exist_and_define_base_and_fla2: OK")
    import torch
    if torch.cuda.is_available():
        test_training_smoke_tree()
        print("test_training_smoke_tree: OK")
    else:
        print("test_training_smoke_tree: skipped (no CUDA)")
    print("All smoke tests passed.")
