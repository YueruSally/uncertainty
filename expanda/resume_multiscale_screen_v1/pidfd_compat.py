"""Use Linux pidfds even when Python was built without their optional bindings.

No PID-based kill fallback. Syscall numbers are restricted to verified 64-bit ABIs:
https://github.com/torvalds/linux/blob/v6.8/arch/x86/entry/syscalls/syscall_64.tbl
https://github.com/torvalds/linux/blob/v6.8/include/uapi/asm-generic/unistd.h
"""
import ctypes
import os
import platform
import signal
import sys


class PidfdAPI:
    def __init__(self):
        if sys.platform != 'linux':
            raise RuntimeError('pidfd handoff requires the Linux experiment server')
        self.native_open = getattr(os,'pidfd_open',None)
        self.native_send = getattr(signal,'pidfd_send_signal',None)
        if callable(self.native_open) and callable(self.native_send):
            self.backend = 'python-pidfd'
            return
        architecture = platform.machine().lower()
        if architecture not in ('x86_64','aarch64') or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError('No verified pidfd compatibility ABI: '+architecture)
        self.libc = ctypes.CDLL(None,use_errno=True)
        self.syscall = self.libc.syscall
        self.syscall.restype = ctypes.c_long
        self.syscall.argtypes = [ctypes.c_long]  # remaining variadic arguments are explicitly typed
        self.backend = 'linux-pidfd-syscall/'+architecture

    def call(self,number,*args):
        ctypes.set_errno(0)
        result = int(self.syscall(ctypes.c_long(number),*args))
        if result == -1:
            error = ctypes.get_errno()
            raise OSError(error,os.strerror(error))
        return result

    def open(self,pid):
        if self.backend == 'python-pidfd':
            return self.native_open(pid,0)
        return self.call(434,ctypes.c_int(pid),ctypes.c_uint(0))

    def send(self,fd,signum):
        if self.backend == 'python-pidfd':
            return self.native_send(fd,signum,None,0)
        return self.call(424,ctypes.c_int(fd),ctypes.c_int(signum),ctypes.c_void_p(),ctypes.c_uint(0))


def self_check():
    """Signal 0 probes only our own process; no task receives a control signal."""
    api = PidfdAPI()
    fd = api.open(os.getpid())
    try:
        api.send(fd,0)
    finally:
        os.close(fd)
    return {'completed':True,'backend':api.backend,'checked_pid':os.getpid(),
            'signal':0,'experiment_processes_changed':False}
