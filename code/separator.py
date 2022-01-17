"""
@file: separator.py
- two major platforms use different folder separator detector / (for Mac OS) and \\ (for Windows)
- this function is built to insert the appropriate separator
@author: Rui Hu (Sherman)
"""

# import platform from sys to detect the operating system
from sys import platform


# define separator - returns appropriate separator
def separator() -> str:
    if platform == "linux" or platform == "linux2":
        return "/"  # Linux
    elif platform == "darwin":
        return "/"  # Mac OS
    elif platform == "win32":
        return "\\"  # Windows
