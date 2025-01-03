import sys
import os
import inspect
import time
import subprocess
from threading import Lock
import threading
import json
import re
from types import FrameType
from typing import Callable, Optional
import platform


class TMLLInstrumentation:
    """
    A module for enabling user-space instrumentation (Python) and kernel instrumentation (LTTng).
    This is an internal module, and in cases where performance is critical, it is recommended to not use this module.
    """

    enabled = False
    _lock = Lock()
    _signature_cache = {}
    _instrumentation_file = None
    _original_instrumentation = None
    _lttng = None

    class _LTTngService:
        """
        A class for managing LTTng instrumentation, which is responsible for kernel-level tracing.
        """

        def __init__(self, session_name: str, output_path: str, verbose: bool = False):
            """
            Initialize the LTTng service.

            :param session_name: The name of the LTTng session
            :type session_name: str
            :param output_path: The output path for the LTTng instrumentations
            :type output_path: str
            :param verbose: Whether to show LTTng command outputs
            :type verbose: bool
            """
            self.session_name = session_name
            self.output_path = output_path
            self.verbose = verbose

        def start(self) -> None:
            # Clean up any existing session
            self.__destroy()

            # Initialize instrumentation
            self.__create_session()
            self.__enable_channel()
            self.__enable_event()

            # Add contexts
            for context in ['tid', 'pid', 'procname']:
                self.__add_context(context)

            # Start instrumentation
            self.__start()

        def stop(self) -> None:
            self.__stop()
            self.__destroy()

        def __create_session(self) -> None:
            subprocess.run(['lttng', 'create', self.session_name, '--output', self.output_path],
                           capture_output=not self.verbose)

        def __enable_channel(self) -> None:
            subprocess.run(['lttng', 'enable-channel', '--kernel', '--subbuf-size=64M',
                            '--num-subbuf=8', f'{self.session_name}-channel'],
                           capture_output=not self.verbose)

        def __enable_event(self) -> None:
            # Enable kernel events (the most common and useful events)
            kernel_events = ('sched_switch,sched_waking,sched_pi_setprio,sched_process_fork,'
                             'sched_process_exit,sched_process_free,sched_wakeup,irq_softirq_entry,'
                             'irq_softirq_raise,irq_softirq_exit,irq_handler_entry,irq_handler_exit,'
                             'lttng_statedump_process_state,lttng_statedump_start,lttng_statedump_end,'
                             'lttng_statedump_network_interface,lttng_statedump_block_device,'
                             'block_rq_complete,block_rq_insert,block_rq_issue,block_bio_frontmerge,'
                             'sched_migrate,sched_migrate_task,power_cpu_frequency,net_dev_queue,'
                             'netif_receive_skb,net_if_receive_skb,timer_hrtimer_start,'
                             'timer_hrtimer_cancel,timer_hrtimer_expire_entry,timer_hrtimer_expire_exit')

            subprocess.run(['lttng', 'enable-event', '--kernel', kernel_events,
                            f'--channel={self.session_name}-channel'],
                           capture_output=not self.verbose)

            # Enable all syscalls
            subprocess.run(['lttng', 'enable-event', '--kernel', '--syscall', '--all',
                            f'--channel={self.session_name}-channel'],
                           capture_output=not self.verbose)

        def __add_context(self, context: str) -> None:
            subprocess.run(['lttng', 'add-context', '--kernel', f'--type={context}',
                            f'--channel={self.session_name}-channel'],
                           capture_output=not self.verbose)

        def __start(self) -> None:
            subprocess.run(['lttng', 'start', self.session_name],
                           capture_output=not self.verbose)

        def __stop(self) -> None:
            subprocess.run(['lttng', 'stop', self.session_name],
                           capture_output=not self.verbose)

        def __destroy(self) -> None:
            subprocess.run(['lttng', 'destroy', self.session_name],
                           capture_output=not self.verbose)

    @classmethod
    def _get_function_signature(cls, frame: FrameType) -> str:
        """
        Get the function signature from the frame.

        :param frame: The frame object
        :type frame: FrameType
        :return: The function signature
        :rtype: str
        """
        code = frame.f_code
        key = f"{code.co_filename}:{code.co_name}"

        cached_sig = cls._signature_cache.get(key)
        if cached_sig is not None:
            return cached_sig

        try:
            func = frame.f_globals.get(code.co_name)
            if not func or not inspect.isfunction(func):
                cls._signature_cache[key] = "()"
                return "()"

            sig = inspect.signature(func)
            params = [param.annotation.__name__ if param.annotation != inspect._empty else 'object'
                      for param in sig.parameters.values()]

            signature = f"({','.join(params)})"
            cls._signature_cache[key] = signature
            return signature
        except:
            cls._signature_cache[key] = "()"
            return "()"

    @classmethod
    def _write_instrumentation(cls, message: str) -> None:
        """
        Write the instrumentation message to the instrumentation file.

        :param message: The instrumentation message
        :type message: str
        """
        if cls._instrumentation_file:
            try:
                with cls._lock:
                    with open(cls._instrumentation_file, 'a', encoding='utf-8') as f:
                        f.write(message + '\n')
                        f.flush()
            except Exception:
                pass

    @classmethod
    def _instrumentation_callback(cls, frame: FrameType, event: str, arg: Optional[str]) -> Optional[Callable]:
        """
        The callback function for instrumentation.

        :param frame: The frame object
        :type frame: FrameType
        :param event: The event type
        :type event: str
        :param arg: The argument
        :type arg: Optional[str]
        :return: The instrumentation callback function
        :rtype: Optional[Callable]
        """
        try:
            if not cls.enabled:
                return cls._original_instrumentation if cls._original_instrumentation else None

            if event not in ('call', 'return'):
                return cls._instrumentation_callback

            module = inspect.getmodule(frame)
            if not module:
                return cls._instrumentation_callback

            code = frame.f_code
            if 'self' in frame.f_locals:
                class_name = frame.f_locals['self'].__class__.__name__
                func_name = f"{module.__name__}.{class_name}.{code.co_name}"
            else:
                func_name = f"{module.__name__}.{code.co_name}"

            signature = cls._get_function_signature(frame)
            func_name = f"{func_name}{signature}"

            message = f"[{time.time_ns()}] [TID:{threading.get_native_id()}] [PID:{os.getpid()}] {'S' if event == 'call' else 'E'} {func_name}"
            cls._write_instrumentation(message)

            if cls._original_instrumentation:
                cls._original_instrumentation(frame, event, arg)

            return cls._instrumentation_callback
        except Exception:
            return cls._instrumentation_callback

    @classmethod
    def convert_to_json(cls, instrumentation_file: str, output_file: Optional[str] = None) -> None:
        """
        Convert the instrumentation file to JSON format.

        :param instrumentation_file: The path to the instrumentation file
        :type instrumentation_file: str
        :param output_file: The path to the output JSON file
        :type output_file: Optional[str]
        """
        if not output_file:
            output_file = os.path.splitext(instrumentation_file)[0] + '.json'

        pattern = r'\[(\d+)\] \[TID:(\d+)\] \[PID:(\d+)\] ([SE]) (.+)'

        try:
            events = []
            with open(instrumentation_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue

                    match = re.match(pattern, line.strip())
                    if match:
                        timestamp_ns, tid, pid, event_type, func_name = match.groups()
                        events.append({
                            "ts": float(timestamp_ns) / 1000.0,
                            "ph": "B" if event_type == "S" else "E",
                            "name": func_name,
                            "pid": int(pid),
                            "tid": int(tid)
                        })

            with open(output_file, 'w') as f:
                json.dump(events, f, indent=2)

        except Exception as e:
            print(f"Error converting instrumentation to JSON: {str(e)}")

    @classmethod
    def enable(cls, instrumentation_file: Optional[str] = None, instrument_kernel: bool = False, verbose: bool = False) -> None:
        """
        Enable both Python function instrumentation and LTTng kernel instrumentation.

        :param instrumentation_file: Where to save the instrumentation file
        :type instrumentation_file: Optional[str]
        :param instrument_kernel: Whether to instrument the kernel (available only on Linux)
        :type instrument_kernel: bool
        :param verbose: Whether to show verbose output for LTTng commands
        :type verbose: bool
        """
        if not cls.enabled:
            if instrumentation_file:
                instrumentation_file = os.path.abspath(instrumentation_file)
            else:
                instrumentation_file = os.path.join(os.getcwd(), f"tmll-{int(time.time())}", "ust.log")

            os.makedirs(os.path.dirname(instrumentation_file), exist_ok=True)
            cls._instrumentation_file = instrumentation_file

            # Start LTTng instrumentation
            if instrument_kernel:
                # Only support Linux for now
                if platform.system().lower() == 'linux':
                    session_name = os.path.basename(instrumentation_file).split('.')[0]
                    lttng_output_path = os.path.dirname(instrumentation_file)

                    cls._lttng = cls._LTTngService(session_name, lttng_output_path, verbose)
                    cls._lttng.start()

            # Start Python function instrumentation (UST)
            cls._original_instrumentation = sys.gettrace()
            cls._signature_cache.clear()
            cls.enabled = True
            sys.settrace(cls._instrumentation_callback)

    @classmethod
    def disable(cls, convert_to_json: bool = False) -> None:
        """
        Disable both Python function instrumentation and LTTng kernel instrumentation.

        :param convert_to_json: Whether to convert the instrumentation file to JSON format
        :type convert_to_json: bool
        """
        if cls.enabled:
            # Stop Python function instrumentation
            cls.enabled = False
            sys.settrace(cls._original_instrumentation)

            if convert_to_json and cls._instrumentation_file:
                cls.convert_to_json(cls._instrumentation_file)

            cls._original_instrumentation = None
            cls._instrumentation_file = None
            cls._signature_cache.clear()

            # Stop LTTng instrumentation
            if cls._lttng:
                cls._lttng.stop()
                cls._lttng = None
