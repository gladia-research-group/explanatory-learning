import os
from pathlib import Path

import dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

def get_env(env_name: str) -> str:
    """
    Read an environment variable.
    Raises errors if it is not defined or empty.

    :param env_name: the name of the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        raise KeyError(f"{env_name} not defined")
    env_value: str = os.environ[env_name]
    if not env_value:
        raise ValueError(f"{env_name} has yet to be configured")
    return env_value


def load_envs(env_file: str = ".env") -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use
    """
    assert os.path.isfile(env_file), f"{env_file}"
    dotenv.load_dotenv(dotenv_path=env_file, override=True)