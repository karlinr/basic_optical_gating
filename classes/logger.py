# General imports
import time

# SECTION:  Colour class
class Colour:
    """_summary_ : Colour class for pretty printing
    """
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
#!SECTION

# SECTION: logger class
class Logger: 
    def __init__(self, name = "Undefined") -> None:
        """ Logger class for pretty printing
            Args:
                name (str, optional): Name of the logger. Defaults to "Undefined".
        """        
        # Set default log levels
        # Errors and warnings are shown by default
        self.last_time = time.time()
        self.set_normal()
        self.name = name

    def set_log_level(self, show_errors = True, show_warnings = True, show_infos = True, show_successes = True, show_debugs = True):
        """ Set the log level for printing messages. Generally we should use the helper functions to set these.

            Args:
                show_errors (bool, optional): Show errors. Defaults to True.
                show_warnings (bool, optional): Show warnings. Defaults to True.
                show_infos (bool, optional): Show infos. Defaults to True.
                show_successes (bool, optional): Show successes. Defaults to True.
                show_debugs (bool, optional): Show debug message and time between messages. Defaults to True.
        """        
        self.log_level = {
            "ERROR": show_errors,
            "WARNING": show_warnings,
            "INFO": show_infos,
            "SUCCESS": show_successes,
            "DEBUG": show_debugs
        }
    
    def set_debug(self):
        """Show all messages"""        
        self.set_log_level(show_errors = True, show_warnings = True, show_infos = True, show_successes = True, show_debugs = True)

    def set_normal(self):
        """Show all messages except debug messages"""
        self.set_log_level(show_errors = True, show_warnings = True, show_infos = True, show_successes = True, show_debugs = False)

    def set_quiet(self):
        """Show only errors and warnings"""
        self.set_log_level(show_errors = True, show_warnings = True, show_infos = False, show_successes = False, show_debugs = False)

    def set_silent(self):
        """Show only errors"""
        self.set_log_level(show_errors = True, show_warnings = False, show_infos = False, show_successes = False, show_debugs = False)

    # Used for timer
    def _set_time(self):
        """Start our timer"""        
        self.last_time = time.time()

    def _get_time(self):
        """ Get the time since the last time we called _set_time()

            Returns:
                float: Time since last call to _set_time()
        """
        return time.time() - self.last_time

    # Display log message
    def print_message(self, type = "INFO", message = "No message"):
        """ Print a message to the console

            Args:
                type (str, optional): Type of message. Choices are "ERROR", "WARNING", "INFO", "SUCCESS", "DEBUG". Defaults to "INFO".
                message (str, optional): Output message. Defaults to "No message".
        """        
        p_name = ""
        p_marker = ""
        printme = False
        
        if self.log_level[type]:
            if type == "ERROR":
                p_name = f"{Colour.RED}{self.name}{Colour.END}"
                p_marker = f"{Colour.RED}\u26a0{Colour.END}"
                printme = True
            elif type == "WARNING":
                p_name = f"{Colour.YELLOW}{self.name}{Colour.END}"
                p_marker = f"{Colour.YELLOW}\u26a0{Colour.END}"
                printme = True
            elif type == "INFO":
                p_name = f"{Colour.BLUE}{self.name}{Colour.END}"
                p_marker = f"{Colour.BLUE}i{Colour.END}"
                printme = True
            elif type == "SUCCESS":
                p_name = f"{Colour.GREEN}{self.name}{Colour.END}"
                p_marker = f"{Colour.GREEN}\u2713{Colour.END}"
                printme = True
            elif type == "DEBUG":
                p_name = f"{Colour.CYAN}{self.name}{Colour.END}"
                p_marker = f"{Colour.CYAN}?{Colour.END}"
                printme = True

        if printme == True:
            if self.log_level["DEBUG"]:
                print(f"{p_name} {p_marker} {message} {Colour.CYAN}({self._get_time():.2f}s){Colour.END}")
                self._set_time()
            else:
                print(f"{p_name} {p_marker} {message}")

    def __repr__(self) -> str:
        return_string = f"Logger({self.name})\n"
        for key, value in self.log_level.items():
            return_string += f"\t{key}:"
            if value:
                return_string += f"{Colour.GREEN} True{Colour.END}\n"
            else:
                return_string += f"{Colour.RED} False{Colour.END}\n"

        return return_string
# !SECTION