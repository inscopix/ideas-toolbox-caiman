import logging
import os
import textwrap
from inspect import getframeinfo, stack
from tabulate import tabulate

logger = logging.getLogger()


def _ideas_error_logger(
    exit_code: str,
    message: str,
    log_file: str = "exit_status.txt",
):
    """log errors raised using IdeasError in file for
    BE to pick up and read"""
    with open(log_file, "w") as file:
        file.write(f"{exit_code} , {message}\n")


class IdeasError(Exception):
    """Exception raised for errors that occur in the tools."""

    def __init__(self, message: str):
        """Initialize ToolException instance."""
        self.message = message

        caller = getframeinfo(stack()[1][0])

        calling_module = os.path.basename(caller.filename)
        calling_module = calling_module.replace(".py", "")

        log_safe_message = message.replace("\r", "").replace("\n", "")

        # remove new lines from the message
        # because ideas-BE requires each message to be on
        # a new line
        _ideas_error_logger(
            f"ERR_{calling_module}_{str(caller.lineno)}",
            log_safe_message,
        )

        message = textwrap.wrap(message)
        message = "\n".join(message)
        message = tabulate([[message]], tablefmt="grid")
        message = "\n" + message

        logger.error(message)

        super().__init__(self.message)
