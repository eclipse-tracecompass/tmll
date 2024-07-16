"""
This module provides the TSP installer class that installs the Trace Server Protocol (TSP) on the system.

Classes:
    TspInstaller: A class to install the Trace Server Protocol (TSP) on the system.

Current (semi) supported systems:
    - Windows
    - Linux
    - MacOS (Will be supported in the future)

W.I.P:
    The TSP installer files are not yet available for downloading from the Eclipse website. The installer will be updated once the files are available.
"""

import platform
import os
import subprocess
import urllib.request

from tmll.common.services.logger import Logger

DOWNLOAD_URL = {
    "Windows": "https://www.eclipse.org/downloads/download.php?file=/tracecompass.incubator/trace-server/rcp/trace-compass-server-latest-win32.win32.x86_64.tar.gz&mirror_id=1135",
    "Linux": "https://www.eclipse.org/downloads/download.php?file=/tracecompass.incubator/trace-server/rcp/trace-compass-server-latest-linux.gtk.x86_64.tar.gz"
}

INSTALL_DIRECTORY = {
    "Windows": "C:\\trace-server-protocol",
    "Linux": "/usr/local/bin/trace-server-protocol"
}


class TspInstaller:
    def __init__(self):
        self.logger = Logger("TspInstaller")

        self.logger.info("Initializing TSP Installer")

    def install(self):
        # Check whether the system is either Windows, Linux, or MacOS
        system = platform.system()
        match system:
            case "Windows":
                self.__install_windows()
            case "Linux":
                self.__install_linux()
            case _:
                self.logger.error(f"Unsupported system: {system}")
                raise Exception(f"Unsupported system: {system}")

    def __install_windows(self) -> None:
        self.logger.info("Installing TSP on Windows")

        # Check whether the installation directory exists
        if not os.path.exists(INSTALL_DIRECTORY["Windows"]):
            os.makedirs(INSTALL_DIRECTORY["Windows"])

        # Check whether the TSP has been installed already
        if not os.path.exists(os.path.join(INSTALL_DIRECTORY["Windows"], "tracecompass-server.exe")):
            # Download the TSP executable
            self.logger.info("Downloading TSP executable")
            urllib.request.urlretrieve(DOWNLOAD_URL["Windows"], "tsp.tar.gz")

            # Extract the TSP executable to the installation directory
            self.logger.info("Extracting TSP executable")
            subprocess.run(["tar", "-xvf", "tsp.tar.gz", "-C", INSTALL_DIRECTORY["Windows"]], shell=True)

            # Remove the downloaded TSP executable
            self.logger.info("Removing downloaded TSP executable")
            subprocess.run(["del", "tsp.tar.gz"], shell=True)

            self.logger.info("TSP installed successfully")

        # Run the TSP server
        self.logger.info("Running TSP server")
        subprocess.Popen([os.path.join(INSTALL_DIRECTORY["Windows"], "tracecompass-server.exe")], shell=True)

        return

    def __install_linux(self):
        pass
