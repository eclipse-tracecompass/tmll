from ast import Name
import random
import string
import hashlib
from typing import Optional


class NameGenerator:
    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    @staticmethod
    def generate_name(base: Optional[str] = None, length: int = 8) -> str:
        """Generate a name. If base is not None, it should create a N-character name based on base. If base is None, it should create a random name.

        Args:
            base (Optional[str], optional): The base name. Defaults to None.
        """

        if base is None:
            return NameGenerator.__generate_random_name(length)
        else:
            return NameGenerator.__generate_name_from_base(base, length)

    @staticmethod
    def __generate_random_name(length: int) -> str:
        """This method should generate a random name with the specified length.

        Args:
            length (int): The length of the name to generate.

        Returns:
            str: The generated name.
        """

        name = random.SystemRandom().choices(string.ascii_letters + string.digits, k=length)
        return "".join(name)

    @staticmethod
    def __generate_name_from_base(base: str, length: int) -> str:
        """This method should generate a name with the specified length based on the base name.

        Args:
            base (str): The base name.
            length (int): The length of the name to generate.

        Returns:
            str: The generated name.
        """

        # Create a hash from the base name with the specified length
        hash = hashlib.sha256(base.encode("utf-8")).hexdigest()
        return hash[:length]
