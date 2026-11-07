"""Run LeRobot recording with keyboard input from the foreground terminal."""

import os
import runpy
import select
import sys
import termios
import threading
import tty


class TerminalKeyboard:
    def __init__(self, events, fd):
        self.events = events
        self.fd = fd
        self.settings = termios.tcgetattr(fd)
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self):
        tty.setcbreak(self.fd)
        self.thread.start()

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=1)
        termios.tcsetattr(self.fd, termios.TCSANOW, self.settings)

    def _read(self):
        while not self.stopped.is_set():
            if not select.select([self.fd], [], [], 0.05)[0]:
                continue
            key = os.read(self.fd, 1)
            if not key:
                return
            if key == b'\x1b':
                # Distinguish standalone Escape from an arrow's escape sequence.
                if select.select([self.fd], [], [], 0.1)[0]:
                    second = os.read(self.fd, 1)
                    if second in (b'[', b'O') and select.select([self.fd], [], [], 0.1)[0]:
                        key += second + os.read(self.fd, 1)
                    else:
                        continue
            if key in (b'\x1b[C', b'\x1bOC', b'n'):
                self.events['exit_early'] = True
                print('\n[KEY] Next stage', flush=True)
            elif key in (b'\x1b[D', b'\x1bOD', b'r'):
                self.events['rerecord_episode'] = True
                self.events['exit_early'] = True
                print('\n[KEY] Re-record episode', flush=True)
            elif key in (b'\x1b', b'q'):
                self.events['stop_recording'] = True
                self.events['exit_early'] = True
                print('\n[KEY] Stop recording', flush=True)
                return


def main():
    from lerobot.utils import control_utils

    original = control_utils.init_keyboard_listener
    listeners = []

    def init_listener():
        if not sys.stdin.isatty():
            return original()
        events = dict(exit_early=False, rerecord_episode=False, stop_recording=False)
        listener = TerminalKeyboard(events, sys.stdin.fileno())
        listener.start()
        listeners.append(listener)
        print('Terminal keys: Right/n = next; Left/r = retry; Esc/q = stop.', flush=True)
        return listener, events

    control_utils.init_keyboard_listener = init_listener
    try:
        runpy.run_module('lerobot.record', run_name='__main__')
    finally:
        for listener in listeners:
            listener.stop()


if __name__ == '__main__':
    main()
