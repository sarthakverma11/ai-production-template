from pathlib import Path

from src.config_loader import load_config
from src.logger import get_logger
from src.utils import ensure_dir, save_text_artifact, seed_everything


def main() -> None:
    config = load_config("configs/config.yaml")

    output_dir = config["runtime"]["output_dir"]
    log_dir = config["runtime"]["log_dir"]
    run_name = config["experiment"]["run_name"]

    ensure_dir(output_dir)
    ensure_dir(log_dir)

    logger = get_logger(
        name="ai-production-template",
        log_dir=log_dir,
        log_file=f"{run_name}.log",
    )

    logger.info("Loaded configuration: %s", config)

    seed = config["runtime"]["seed"]
    seed_everything(seed)
    logger.info("Random seed set to %s", seed)

    project_name = config["project"]["name"]
    environment = config["project"]["environment"]
    model_name = config["model"]["name"]
    temperature = config["model"]["temperature"]

    # This is a simulated AI response. No real model or API is called.
    response = (
        f"Project '{project_name}' is running in '{environment}' environment "
        f"using model '{model_name}' with temperature={temperature}."
    )

    artifact_path = Path(output_dir) / "run_summary.txt"
    save_text_artifact(str(artifact_path), response)

    logger.info("Generated artifact: %s", artifact_path)
    logger.info("Project run completed successfully.")
    print("Project run completed successfully.")


if __name__ == "__main__":
    main()
