import sys
import logging
import traceback  # Import the traceback module

def error_message_details(error, error_detail=None):
    """
    Formats the error message with details, including filename and line number if available.

    Args:
        error: The exception object.
        error_detail: Optional.  The exception info from sys.exc_info().
    Returns:
        str: A formatted error message.
    """
    if error_detail:
        try:
            _, _, exc_tb = error_detail
            filename = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_frame.f_lineno
            error_message = (
                f"Error occurred in python script name [{filename}] "
                f"line number [{line_number}] error message [{str(error)}]"
            )
        except AttributeError:  # Handle exceptions without standard traceback
            error_message = f"Error occurred: {str(error)}\nTraceback:\n{traceback.format_exc()}"
    else:
        error_message = f"Error occurred: {str(error)}"
    return error_message


class CustomException(Exception):
    """
    Custom exception class to provide detailed error messages.
    """
    def __init__(self, error_message, error_detail=None):
        """
        Initializes the CustomException.

        Args:
            error_message: The error message string.
            error_detail: Optional.  The exception info from sys.exc_info().
        """
        super().__init__(error_message)  # Call the base class constructor
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        """
        Returns the formatted error message when the exception is printed.
        """
        return self.error_message



if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    try:
        a = 1 / 0
    except Exception as e:
        logging.error("Division by zero error", exc_info=True)
        raise CustomException(e, sys.exc_info()) from e
