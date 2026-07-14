"""
Password Hashing Utility

Provides functions for hashing passwords and verifying them
using the bcrypt algorithm.
"""

from passlib.context import CryptContext

# Create a single CryptContext object to be used throughout the application.
# We use bcrypt as the standard strong hashing algorithm.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against its hashed version.

    Args:
        plain_password: The unhashed password provided by the user.
        hashed_password: The hashed password retrieved from the database.

    Returns:
        True if the passwords match, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a plain text password using bcrypt.

    Args:
        password: The plain text password to hash.

    Returns:
        The bcrypt hashed password string.
    """
    return pwd_context.hash(password)
