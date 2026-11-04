"""parallelize(task, ctx, items, nthreads): run task(ctx, i) for every i in 0..items on `nthreads` pthreads, items
handed out by an atomic counter; returns when all are done. `ctx` is the address of whatever the task reads
(a struct on the caller's stack). Plain pthreads, so a build needs the `mojo` package only."""
from std.atomic import Atomic
from std.ffi import external_call
from std.sys import num_logical_cores

comptime Task = def(Int, Int) thin -> None

@fieldwise_init
struct _Job:
    var task: Task
    var ctx: Int
    var next: Pointer[Int64, MutUntrackedOrigin]
    var items: Int

def _run(arg: Int) abi("C") -> Int:
    var job = Pointer[_Job, MutUntrackedOrigin](unsafe_from_address=arg)
    while True:
        var i = Int(Atomic[Int64].fetch_add(job[].next, Int64(1)))
        if i >= job[].items:
            return 0
        job[].task(job[].ctx, i)

def parallelize(task: Task, ctx: Int, items: Int, nthreads: Int = 0):
    var n = min(items, nthreads if nthreads > 0 else num_logical_cores())
    if n <= 1:
        for i in range(items):
            task(ctx, i)
        return
    var counter = List[Int64](length=1, fill=0)
    var job = _Job(task, ctx, Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(counter.unsafe_ptr())), items)
    var jp = Int(Pointer(to=job))
    var tids = List[UInt64](length=n - 1, fill=0)
    var started = 0
    for t in range(n - 1):
        if external_call["pthread_create", Int32](tids.unsafe_ptr().unsafe_offset(t), Int(0), _run, jp) != 0:
            break
        started += 1
    _ = _run(jp)  # the caller works too, so a failed pthread_create only costs speed
    for t in range(started):
        _ = external_call["pthread_join", Int32](tids[t], Int(0))
    _ = len(counter) + job.items
