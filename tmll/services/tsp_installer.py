"""
This module provides the TSP installer class that installs the Trace Server Protocol (TSP) on the system.

Classes:
    TSPInstaller: A class to install the Trace Server Protocol (TSP) on the system.

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
import shutil

from tmll.common.services.logger import Logger

DOWNLOAD_URL = {
    "Windows": "https://download.eclipse.org/tracecompass.incubator/trace-server/rcp/trace-compass-server-latest-win32.win32.x86_64.tar.gz",
    "Linux": "https://download.eclipse.org/tracecompass.incubator/trace-server/rcp/trace-compass-server-latest-linux.gtk.x86_64.tar.gz",
    "MacOS": "https://download.eclipse.org/tracecompass.incubator/trace-server/rcp/trace-compass-server-latest-macosx.cocoa.x86_64.tar.gz"
}

"""
The installation directory for the TSP server. The installation directory is different for each system.
Directories:
    - Windows: C:\\trace-server-protocol
    - Linux: /usr/local/bin/trace-server-protocol
    - MacOS: /usr/local/bin/trace-server-protocol
"""
INSTALL_DIRECTORY = {
    "Windows": os.path.join("C:\\", "trace-server-protocol"),
    "Linux": os.path.join("/", "usr", "local", "bin", "trace-server-protocol"),
    "MacOS": os.path.join("/", "usr", "local", "bin", "trace-server-protocol")
}


class TSPInstaller:
    def __init__(self):
        self.logger = Logger("TSPInstaller")

        self.logger.info("Initializing TSP Installer")

    def install(self):
        # Check whether the system is either Windows, Linux, or MacOS
        system = platform.system()
        match system:
            case "Windows":
                self.__install("Windows", "tracecompass-server.exe")
            case "Linux":
                self.__install("Linux", "tracecompass-server")
            case "Darwin":
                self.__install("MacOS", "tracecompass-server")
            case _:
                self.logger.error(f"Unsupported system: {system}")
                raise Exception(f"Unsupported system: {system}")

    def __install(self, os_name: str, executable_name: str) -> None:
        self.logger.info(f"Installing TSP on {os_name}")

        install_dir = INSTALL_DIRECTORY[os_name]
        download_url = DOWNLOAD_URL[os_name]

        # Check whether the installation directory exists
        if not os.path.exists(install_dir):
            os.makedirs(install_dir)

        # Check whether the TSP has been installed already
        if not os.path.exists(os.path.join(install_dir, executable_name)):
            # Download the TSP executable
            self.logger.info("Downloading TSP executable")
            subprocess.run(["curl", "-o", "tsp.tar.gz", download_url], capture_output=True)

            # Extract the TSP executable to the installation directory
            self.logger.info("Extracting TSP executable")
            subprocess.run(["tar", "-xvf", "tsp.tar.gz", "-C", install_dir], capture_output=True)

            # Move the contents of the extracted folder to the installation directory
            self.logger.info("Moving TSP executable to the installation directory")
            extracted_dir = os.path.join(install_dir, "trace-compass-server")
            for file in os.listdir(extracted_dir):
                shutil.move(os.path.join(extracted_dir, file), install_dir)
            os.rmdir(extracted_dir)

            # Remove the downloaded TSP executable
            self.logger.info("Removing downloaded TSP executable")
            if os_name == "Windows":
                subprocess.run(["del", "tsp.tar.gz"], shell=True, capture_output=True)
            else:
                subprocess.run(["rm", "tsp.tar.gz"], capture_output=True)

            self.logger.info("TSP installed successfully")

        # Run the TSP server
        self.logger.info("Running TSP server")
        if os_name == "Windows":
            subprocess.Popen([os.path.join(install_dir, executable_name)], shell=True)
        else:
            subprocess.Popen([os.path.join(install_dir, executable_name)])
