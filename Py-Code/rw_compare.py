#!/usr/bin/env python3
"""
rw_compare.py - Readers-Writers problem: four synchronisation methods, tested and plotted.

Methods compared
  1. Semaphore  - classic "first readers-writers" solution (reader preference)
  2. Monitor    - lock + condition variables (writer preference, Mesa semantics)
  3. Dekker     - Dekker's 2-process lock, composed into a tournament for N threads
  4. Peterson   - Peterson's 2-process lock, composed into a tournament for N threads

What the program does
  * runs every method with the same workload (readers, writers, iterations),
  * checks the safety invariant (no reader with a writer, never two writers),
  * measures waiting time per access and total time,
  * draws a Gantt chart (one panel per method) and saves it as a PNG.

Usage examples
  python3 rw_compare.py                          # 3 readers, 2 writers, 3 iterations
  python3 rw_compare.py -r 4 -w 2 -i 5 -o g.png  # custom workload
  python3 rw_compare.py --no-plot                # table only (no matplotlib needed)

Important educational note about Python
  CPython has a Global Interpreter Lock (GIL), so only one thread runs bytecode at a
  time and our plain-variable Dekker/Peterson code behaves as if memory were
  sequentially consistent. In C/Java on real multicore hardware the same algorithms
  need atomics/fences (see the paper). Here busy waiting uses time.sleep(0) to yield
  the GIL; without it a spinning thread would starve the lock holder.
"""
import argparse
import threading
import time

# --------------------------------------------------------------------------------------
# 1. Shared resource + instrumentation
# --------------------------------------------------------------------------------------
class Resource:
    """The shared object. It also CHECKS the safety rule while threads use it."""

    def __init__(self):
        self.value = 0                     # the data being read / written
        self._chk = threading.Lock()       # protects ONLY the checker's counters
        self.active_r = 0
        self.active_w = 0
        self.violations = 0

    def enter(self, is_writer):
        with self._chk:
            if is_writer:
                # a writer must be alone
                if self.active_w or self.active_r:
                    self.violations += 1
                self.active_w += 1
            else:
                # a reader may share with readers but never with a writer
                if self.active_w:
                    self.violations += 1
                self.active_r += 1

    def leave(self, is_writer):
        with self._chk:
            if is_writer:
                self.active_w -= 1
            else:
                self.active_r -= 1


class Trace:
    """Thread-safe log of (role, id, request, enter, exit) times in seconds."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.rows = []
        self._lock = threading.Lock()

    def now(self):
        return time.perf_counter() - self.t0

    def add(self, role, ident, req, ent, ext):
        with self._lock:
            self.rows.append((role, ident, req, ent, ext))


# --------------------------------------------------------------------------------------
# 2. Method 1: Semaphores (readers preference)
# --------------------------------------------------------------------------------------
class SemaphoreRW:
    name = "Semaphore"

    def __init__(self):
        self.mutex = threading.Semaphore(1)  # protects read_count
        self.wrt = threading.Semaphore(1)    # guards the resource (writers / reader group)
        self.read_count = 0

    def start_read(self, tid):
        self.mutex.acquire()                 # wait(mutex)
        self.read_count += 1
        if self.read_count == 1:             # the FIRST reader locks out writers
            self.wrt.acquire()               # wait(wrt)
        self.mutex.release()                 # signal(mutex)

    def end_read(self, tid):
        self.mutex.acquire()
        self.read_count -= 1
        if self.read_count == 0:             # the LAST reader lets writers in
            self.wrt.release()               # signal(wrt)
        self.mutex.release()

    def start_write(self, tid):
        self.wrt.acquire()

    def end_write(self, tid):
        self.wrt.release()


# --------------------------------------------------------------------------------------
# 3. Method 2: Monitor (lock + condition variables, writer preference)
# --------------------------------------------------------------------------------------
class MonitorRW:
    name = "Monitor"

    def __init__(self):
        self.lock = threading.Lock()                   # the monitor's implicit lock
        self.ok_read = threading.Condition(self.lock)  # readers wait here
        self.ok_write = threading.Condition(self.lock)  # writers wait here
        self.active_readers = 0
        self.waiting_writers = 0
        self.writing = False

    def start_read(self, tid):
        with self.lock:
            # Mesa semantics: always re-test the condition in a WHILE loop.
            while self.writing or self.waiting_writers > 0:
                self.ok_read.wait()
            self.active_readers += 1

    def end_read(self, tid):
        with self.lock:
            self.active_readers -= 1
            if self.active_readers == 0:
                self.ok_write.notify()

    def start_write(self, tid):
        with self.lock:
            self.waiting_writers += 1
            while self.writing or self.active_readers > 0:
                self.ok_write.wait()
            self.waiting_writers -= 1
            self.writing = True

    def end_write(self, tid):
        with self.lock:
            self.writing = False
            if self.waiting_writers > 0:
                self.ok_write.notify()       # prefer a waiting writer
            else:
                self.ok_read.notify_all()    # otherwise release all readers


# --------------------------------------------------------------------------------------
# 4. Methods 3 and 4: Dekker / Peterson two-process locks + tournament tree
# --------------------------------------------------------------------------------------
def spin():
    """One iteration of busy waiting: yield the GIL so others can make progress."""
    time.sleep(0)


class PetersonLock:
    """Peterson's algorithm for exactly TWO processes, ids 0 and 1."""

    def __init__(self):
        self.flag = [False, False]   # flag[i]: process i is interested
        self.turn = 0                # who must yield on conflict

    def lock(self, me):
        other = 1 - me
        self.flag[me] = True         # 1) announce interest
        self.turn = other            # 2) be polite: give the turn away
        while self.flag[other] and self.turn == other:   # 3) busy wait
            spin()

    def unlock(self, me):
        self.flag[me] = False


class DekkerLock:
    """Dekker's algorithm for exactly TWO processes, ids 0 and 1."""

    def __init__(self):
        self.wants = [False, False]  # wants[i]: process i wants to enter
        self.turn = 0                # who has priority in a conflict

    def lock(self, me):
        other = 1 - me
        self.wants[me] = True
        while self.wants[other]:             # conflict: the other wants in too
            if self.turn != me:              # not my turn -> back off
                self.wants[me] = False
                while self.turn != me:       # wait until the turn is mine
                    spin()
                self.wants[me] = True

    def unlock(self, me):
        self.turn = 1 - me                   # hand the turn over
        self.wants[me] = False


class TournamentLock:
    """
    N-thread lock built from a binary tree of 2-process locks.
    Leaf i is at heap index LEAVES+i; a thread climbs from its leaf to the root,
    winning one 2-process lock per level (side = which child it comes from).
    """
    LEAVES = 8

    def __init__(self, lock_cls):
        self.nodes = [lock_cls() for _ in range(self.LEAVES)]   # nodes 1..7 are used

    def _path(self, slot):
        n, path = self.LEAVES + slot, []
        while n > 1:
            path.append((n >> 1, n & 1))     # (parent node, side 0/1)
            n >>= 1
        return path

    def lock(self, slot):
        for node, side in self._path(slot):  # bottom -> top
            self.nodes[node].lock(side)

    def unlock(self, slot):
        for node, side in reversed(self._path(slot)):   # top -> bottom
            self.nodes[node].unlock(side)


class TournamentRW:
    """
    Readers-Writers on top of a tournament lock (Dekker or Peterson inside).
    The classic 2-process algorithms are NOT a Readers-Writers solution by
    themselves: here they only replace wait(wrt)/signal(wrt). The reader-group
    logic (read_count, first/last reader) is still needed.
    Writers use slots 0..W-1; the whole reader group uses slot 7.
    """
    READER_SLOT = 7

    def __init__(self, lock_cls, name):
        self.name = name
        self.gate = TournamentLock(lock_cls)
        self.read_count = 0
        # Guard for read_count. Honest simplification: a normal lock, not Dekker/Peterson.
        self.cnt_guard = threading.Lock()

    def start_read(self, tid):
        with self.cnt_guard:
            self.read_count += 1
            if self.read_count == 1:
                self.gate.lock(self.READER_SLOT)   # first reader locks for the group

    def end_read(self, tid):
        with self.cnt_guard:
            self.read_count -= 1
            if self.read_count == 0:
                self.gate.unlock(self.READER_SLOT)  # last reader unlocks

    def start_write(self, tid):
        self.gate.lock(tid)

    def end_write(self, tid):
        self.gate.unlock(tid)


METHODS = [
    lambda: SemaphoreRW(),
    lambda: MonitorRW(),
    lambda: TournamentRW(DekkerLock, "Dekker"),
    lambda: TournamentRW(PetersonLock, "Peterson"),
]

# --------------------------------------------------------------------------------------
# 5. Workload and measurement
# --------------------------------------------------------------------------------------
READ_TIME, WRITE_TIME, THINK_TIME = 0.004, 0.004, 0.002   # seconds (simulated work)


def reader_thread(impl, res, tr, tid, iters):
    for _ in range(iters):
        req = tr.now()
        impl.start_read(tid)                 # ---- entry protocol
        ent = tr.now()
        res.enter(False)
        _ = res.value                        # the "read"
        time.sleep(READ_TIME)
        res.leave(False)
        ext = tr.now()
        impl.end_read(tid)                   # ---- exit protocol
        tr.add("R", tid, req, ent, ext)
        time.sleep(THINK_TIME)               # remainder section


def writer_thread(impl, res, tr, tid, iters):
    for _ in range(iters):
        req = tr.now()
        impl.start_write(tid)
        ent = tr.now()
        res.enter(True)
        res.value += 1                       # the "write"
        time.sleep(WRITE_TIME)
        res.leave(True)
        ext = tr.now()
        impl.end_write(tid)
        tr.add("W", tid, req, ent, ext)
        time.sleep(THINK_TIME)


def run_method(factory, readers, writers, iters):
    impl, res, tr = factory(), Resource(), Trace()
    threads = [threading.Thread(target=reader_thread, args=(impl, res, tr, i, iters)) for i in range(readers)]
    threads += [threading.Thread(target=writer_thread, args=(impl, res, tr, i, iters)) for i in range(writers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    waits = {"R": [], "W": []}
    for role, _, req, ent, _ in tr.rows:
        waits[role].append(ent - req)
    avg = lambda xs: 1000 * sum(xs) / len(xs) if xs else float("nan")
    return dict(name=impl.name, total=total, wait_r=avg(waits["R"]), wait_w=avg(waits["W"]),
                violations=res.violations, ok=(res.value == writers * iters), trace=sorted(tr.rows, key=lambda r: r[2]))


# --------------------------------------------------------------------------------------
# 6. Gantt chart
# --------------------------------------------------------------------------------------
def draw_gantt(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    GREEN, AMBER, GREY = "#007A55", "#C77D00", "#cfd8d4"
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.6 * len(results)))
    for ax, r in zip(axes, results):
        rows = sorted({(role, i) for role, i, *_ in r["trace"]}, key=lambda k: (k[0] != "R", k[1]))
        ypos = {k: n for n, k in enumerate(rows)}
        for role, ident, req, ent, ext in r["trace"]:
            y = ypos[(role, ident)]
            ax.barh(y, ent - req, left=req, height=0.5, color=GREY, hatch="///", edgecolor="#9aa7a2", lw=0.3)  # waiting
            ax.barh(y, ext - ent, left=ent, height=0.5, color=GREEN if role == "R" else AMBER)               # inside resource
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{role}{i + 1}" for role, i in rows])
        ax.set_title(f"{r['name']}: total {r['total']*1000:.0f} ms, mean wait R {r['wait_r']:.1f} ms / W {r['wait_w']:.1f} ms, "
                     f"violations {r['violations']}", fontsize=9, loc="left")
        ax.set_xlabel("time (s)")
        ax.grid(axis="x", alpha=0.3)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GREEN), plt.Rectangle((0, 0), 1, 1, color=AMBER),
                            plt.Rectangle((0, 0), 1, 1, color=GREY, hatch="///")],
                   labels=["Reading (shared)", "Writing (exclusive)", "Waiting"], ncol=3, fontsize=8, loc="lower center")
    fig.tight_layout(rect=(0, 0.04, 1, 1))   # leave room for the legend
    fig.savefig(path, dpi=150)
    print(f"Gantt chart saved to {path}")


# --------------------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compare four Readers-Writers solutions and draw a Gantt chart.")
    ap.add_argument("-r", "--readers", type=int, default=3)
    ap.add_argument("-w", "--writers", type=int, default=2, help="max 7 (tournament has 8 leaves, one is the reader group)")
    ap.add_argument("-i", "--iters", type=int, default=3)
    ap.add_argument("-o", "--out", default="gantt.png")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    assert 1 <= a.writers <= 7, "writers must be in 1..7"

    results = [run_method(f, a.readers, a.writers, a.iters) for f in METHODS]
    print(f"\nWorkload: {a.readers} readers, {a.writers} writers, {a.iters} iterations each\n")
    print(f"{'Method':<10}{'Total(ms)':>10}{'WaitR(ms)':>11}{'WaitW(ms)':>11}{'Violations':>12}{'NoLostUpdate':>14}")
    for r in results:
        print(f"{r['name']:<10}{r['total']*1000:>10.1f}{r['wait_r']:>11.2f}{r['wait_w']:>11.2f}{r['violations']:>12}{str(r['ok']):>14}")
    if not a.no_plot:
        draw_gantt(results, a.out)


if __name__ == "__main__":
    main()
