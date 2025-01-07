from typing import Tuple
import re


class Formatter:
    """
    A class for formatting data into human-readable strings.
    """

    @staticmethod
    def format_bytes(bytes: float) -> Tuple[float, str]:
        """
        Convert bytes to human-readable format with appropriate unit.

        :param bytes: Value in bytes
        :type bytes: float
        :return: Tuple of (converted value, unit)
        :rtype: Tuple[float, str]
        """
        units = ["B", "KB", "MB", "GB", "TB"]
        unit_idx = 0
        value = float(bytes)

        while value >= 1024 and unit_idx < len(units) - 1:
            value /= 1024
            unit_idx += 1

        return value, units[unit_idx]

    @staticmethod
    def format_seconds(time_in_seconds: float) -> Tuple[float, str]:
        """
        Convert seconds to a human-readable string with appropriate units.

        :param time_in_seconds: The time in seconds
        :type time_in_seconds: float
        :return: Tuple of (converted value, unit)
        :rtype: Tuple[float, str]
        """
        units = ["s", "m", "h", "d"]
        thresholds = [1, 60, 3600, 86400]
        time = abs(time_in_seconds)

        # If time is less than 1 second, convert to smaller units
        if time < 1:
            if time < 0.000001:  # nanoseconds
                return time * 1e9, "ns"
            elif time < 0.001:  # microseconds
                return time * 1e6,  "us"
            else:  # milliseconds
                return time * 1e3, "ms"

        # If time is greater than 1 second, convert to larger units
        for i in range(len(units) - 1, -1, -1):
            if time >= thresholds[i]:
                return time / thresholds[i], units[i]

        return time, "s"  # Default to seconds if no other unit is applicable

    @staticmethod
    def parse_time_to_seconds(time_str: str) -> float:
        """
        Parse a time string into proper format.

        :param time_str: Time string (e.g., '1s', '500ms', '24h', '7d')
        :type time_str: str
        :return: Time in seconds as float (e.g., 1.0, 0.5, 24.0, 604800.0)
        :rtype: float
        """
        pattern = r'^(\d+(?:\.\d+)?)(ns|us|ms|s|m|h|d)$'
        match = re.match(pattern, time_str)

        if not match:
            return 1.0

        value, unit = float(match.group(1)), match.group(2)

        conversion = {
            'ns': 1e-9,
            'us': 1e-6,
            'ms': 1e-3,
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400
        }

        return value * conversion[unit]
