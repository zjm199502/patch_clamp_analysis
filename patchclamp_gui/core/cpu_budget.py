import os


def available():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if kernel.GetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.byref(process_mask), ctypes.byref(system_mask)) and process_mask.value:
            return process_mask.value.bit_count()
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def budget(value, capacity):
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = 0
    return min(capacity, requested) if requested > 0 else max(1, capacity * 7 // 10)


def memory_lanes(cpu_limit, largest_bytes):
    try:
        import psutil
        free = psutil.virtual_memory().available
    except (ImportError, OSError):
        if os.name == "nt":
            import ctypes
            class Memory(ctypes.Structure):
                _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                    (name, ctypes.c_uint64) for name in ("total", "free", "page", "page_free", "virtual", "virtual_free", "extra")]
            memory = Memory()
            memory.length = ctypes.sizeof(memory)
            free = memory.free if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)) else 0
        else:
            try:
                free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, AttributeError):
                free = 0
    return max(1, min(cpu_limit, int(free * .5) // max(64 << 20, largest_bytes * 32)))
