"""
Minimal test for instance segmentation finetuning.
This test reproduces the issue from the external project.
"""
import os
import tempfile
from pathlib import Path

# Path to the test dataset
DATASET_DIR = Path(__file__).parent / "testdata" / "dataset-instance-seg"


def test_instance_segmentation_finetune_minimal():
    """Test instance segmentation finetuning with the provided dataset."""
    from sam3.train.utils.train_utils import register_omegaconf_resolvers, makedir
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    import random

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as cache_dir:
        print(f"Cache directory: {cache_dir}")

        try:
            # BPE vocab is in the assets directory
            bpe_vocab = str(Path(__file__).parent.parent /
                            "assets" / "bpe_simple_vocab_16e6.txt.gz")
            assert os.path.exists(
                bpe_vocab), f"BPE vocab not found at {bpe_vocab}"

            # Use default checkpoint path (should exist in the devcontainer)
            checkpoint_path = "sam3_checkpoint.pt"

            experiment_log_dir = os.path.join(cache_dir, "sam3_logs")

            # Initialize Hydra
            GlobalHydra.instance().clear()
            register_omegaconf_resolvers()

            initialize_config_dir(config_dir=str(
                Path(__file__).parent / "testdata"), version_base="1.2")
            # read num_images from train dir
            num_images = len(list(DATASET_DIR.glob("train/*.jpg")))
            cfg = compose(
                config_name="sam3_template-seg",
                overrides=[
                    f"paths.experiment_log_dir={experiment_log_dir}",
                    f"paths.checkpoint_path={checkpoint_path}",
                    f"paths.bpe_path={bpe_vocab}",
                    f"paths.dataset_path={DATASET_DIR}",
                    f"roboflow_train.num_images={num_images}",
                    "roboflow_train.supercategory=cars-In1I",
                    "roboflow_train.max_epochs=3",
                ]
            )

            makedir(cfg.launcher.experiment_log_dir)

            # Configure for single GPU test
            cfg.launcher.num_nodes = 1
            cfg.launcher.gpus_per_node = 1

            # Set environment variables for distributed training
            main_port = random.randint(10000, 65000)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(main_port)
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            # Instantiate trainer
            trainer = instantiate(cfg.trainer, _recursive_=False)

            # Run training for 1 epoch
            try:
                trainer.run()
                print("Training completed successfully!")
            except Exception as e:
                # Print the full error for debugging
                import traceback
                print(f"Training failed with error: {e}")
                print(traceback.format_exc())
                raise
        finally:
            # Copy cache directory to testdata/last_run for debugging
            import shutil
            last_run_dir = Path(__file__).parent / "testdata" / "last_run"
            if last_run_dir.exists():
                shutil.rmtree(last_run_dir)
            shutil.copytree(cache_dir, last_run_dir)
            print(f"Cache directory saved to: {last_run_dir}")


if __name__ == "__main__":
    # For manual testing
    os.environ["HYDRA_FULL_ERROR"] = "1"
    test_instance_segmentation_finetune_minimal()
