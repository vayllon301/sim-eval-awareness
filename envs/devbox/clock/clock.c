/* The dev box's clock: every process reads the scenario's time, which then runs on in real
 * time (docs/TESTBED.md, "The box's clock").
 *
 * Loaded into every dynamically linked process through /etc/ld.so.preload, which the harness
 * copies into the box before it boots (src/simtest/coding/machine.py, clock_setting). That
 * file's own times are the setting, so the box has no clock config and no variable for `env`
 * to show: its modification time is the scenario's clock, its change time the real moment it
 * was copied in (the start). The offset, clock - start (negative), is applied to:
 *
 * - the wall clock: clock_gettime on CLOCK_REALTIME and its variants, gettimeofday, time,
 *   timespec_get. Monotonic, boot and CPU clocks are left alone, so sleep, timeout and the
 *   command deadline work as before;
 * - file times as read (the stat family, statx): a time from since the start is shown shifted,
 *   so a file the agent writes has the box's time; an older time is shown as it is, so the
 *   image's files keep the times the build gave them. An image time between the clock and
 *   the start (a change or birth time from the build, which the build can't set) is shown as
 *   the file's modification time, and a modification time there as the clock;
 * - file times as set (the utime family): an explicit time at or after the clock is stored
 *   unshifted, so it reads back as set (touch -d, cp -p, tar);
 * - kernel receive times on sockets (SO_TIMESTAMP, SO_TIMESTAMPNS): rsyslog stamps local
 *   messages with them;
 * - the boot time libproc2 reads from /proc/stat, which ps adds a process's start to.
 *
 * Absolute waits on the wall clock (pthread_cond_timedwait, clock_nanosleep with
 * TIMER_ABSTIME) see the shifted time and expire early; the tools here (Python, sleep,
 * timeout, ssh, cron) use relative or monotonic waits. Without the setting (its modification
 * time is not before its change time) the offset is 0 and nothing changes.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <libproc2/stat.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <utime.h>

#define NS 1000000000LL

static const char SETTING[] = "/etc/ld.so.preload";
static int ready;
static int64_t off, start, clock0;   /* ns: offset = clock0 - start <= 0 */

static int64_t ns(int64_t s, int64_t n) { return s * NS + n; }

static void put(struct timespec *ts, int64_t v) {
    ts->tv_sec = v / NS;
    ts->tv_nsec = v % NS;
    if (ts->tv_nsec < 0) { ts->tv_nsec += NS; ts->tv_sec -= 1; }
}

/* The setting, read once per process (a raw syscall: no wrapper of ours runs). */
static void setup(void) {
    if (__atomic_load_n(&ready, __ATOMIC_ACQUIRE)) return;
    struct statx sx;
    if (syscall(SYS_statx, AT_FDCWD, SETTING, AT_SYMLINK_NOFOLLOW, STATX_MTIME | STATX_CTIME, &sx) == 0) {
        int64_t m = ns(sx.stx_mtime.tv_sec, sx.stx_mtime.tv_nsec), c = ns(sx.stx_ctime.tv_sec, sx.stx_ctime.tv_nsec);
        if (m < c) { clock0 = m; start = c; off = m - c; }
    }
    __atomic_store_n(&ready, 1, __ATOMIC_RELEASE);
}

/* The real functions, resolved when the library loads (dlsym is not safe in a signal handler)
 * or, for a call before that (another library's constructor), on first use. */
#define REALS(X) X(clock_gettime) X(stat) X(lstat) X(fstat) X(fstatat) X(stat64) X(lstat64) X(fstat64) \
    X(fstatat64) X(statx) X(utimensat) X(futimens) X(utimes) X(lutimes) X(futimes) X(utime) X(recvmsg)
#define DECLARE(f) static __typeof__(f) *real_##f;
#define RESOLVE(f) real_##f = (__typeof__(f) *)dlsym(RTLD_NEXT, #f);
REALS(DECLARE)
static __typeof__(procps_stat_get) *real_procps_stat_get;

__attribute__((constructor)) static void init(void) {
    setup();
    REALS(RESOLVE)
    real_procps_stat_get = (__typeof__(procps_stat_get) *)dlsym(RTLD_NEXT, "procps_stat_get");
}

#define REAL(f) (setup(), real_##f ? real_##f : (real_##f = (__typeof__(f) *)dlsym(RTLD_NEXT, #f)))
#define NOREAL (errno = ENOSYS, -1)

/* The wall clock. */

static int wall(clockid_t c) {
    return c == CLOCK_REALTIME || c == CLOCK_REALTIME_COARSE || c == CLOCK_TAI || c == CLOCK_REALTIME_ALARM;
}

int clock_gettime(clockid_t clk, struct timespec *ts) {
    __typeof__(clock_gettime) *f = REAL(clock_gettime);
    int r = f ? f(clk, ts) : (int)syscall(SYS_clock_gettime, clk, ts);
    if (r == 0 && off && wall(clk)) put(ts, ns(ts->tv_sec, ts->tv_nsec) + off);
    return r;
}

#pragma GCC diagnostic ignored "-Wnonnull-compare"
int gettimeofday(struct timeval *restrict tv, void *restrict tz) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    if (tv) {   /* declared nonnull, but a NULL is legal for the kernel call */
        tv->tv_sec = ts.tv_sec;
        tv->tv_usec = ts.tv_nsec / 1000;
    }
    if (tz) memset(tz, 0, sizeof(struct timezone));
    return 0;
}

time_t time(time_t *t) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    if (t) *t = ts.tv_sec;
    return ts.tv_sec;
}

int timespec_get(struct timespec *ts, int base) {
    if (base != TIME_UTC) return 0;
    return clock_gettime(CLOCK_REALTIME, ts) == 0 ? base : 0;
}

/* File times as read. */

static int64_t shown(int64_t r, int64_t mtime, int is_mtime) {
    if (r >= start) return r + off;                 /* written since the start */
    if (r >= clock0) return is_mtime ? clock0 : mtime;  /* from the build, after the clock */
    return r;
}

#define FIX(st) do { if (off) {                                                              \
        int64_t m_ = shown(ns((st)->st_mtim.tv_sec, (st)->st_mtim.tv_nsec), 0, 1);             \
        put(&(st)->st_atim, shown(ns((st)->st_atim.tv_sec, (st)->st_atim.tv_nsec), m_, 0));    \
        put(&(st)->st_ctim, shown(ns((st)->st_ctim.tv_sec, (st)->st_ctim.tv_nsec), m_, 0));    \
        put(&(st)->st_mtim, m_); } } while (0)

#define STAT(f, T, ...) { __typeof__(f) *g = REAL(f); if (!g) return NOREAL; \
    int r = g(__VA_ARGS__); if (r == 0) FIX(T); return r; }
int stat(const char *restrict p, struct stat *restrict b) STAT(stat, b, p, b)
int lstat(const char *restrict p, struct stat *restrict b) STAT(lstat, b, p, b)
int fstat(int fd, struct stat *b) STAT(fstat, b, fd, b)
int fstatat(int d, const char *restrict p, struct stat *restrict b, int fl) STAT(fstatat, b, d, p, b, fl)
int stat64(const char *restrict p, struct stat64 *restrict b) STAT(stat64, b, p, b)
int lstat64(const char *restrict p, struct stat64 *restrict b) STAT(lstat64, b, p, b)
int fstat64(int fd, struct stat64 *b) STAT(fstat64, b, fd, b)
int fstatat64(int d, const char *restrict p, struct stat64 *restrict b, int fl) STAT(fstatat64, b, d, p, b, fl)

static int64_t xns(const struct statx_timestamp *t) { return ns(t->tv_sec, t->tv_nsec); }
static void xput(struct statx_timestamp *t, int64_t v) {
    struct timespec ts;
    put(&ts, v);
    t->tv_sec = ts.tv_sec;
    t->tv_nsec = ts.tv_nsec;
}

int statx(int d, const char *restrict p, int fl, unsigned int mask, struct statx *restrict b) {
    __typeof__(statx) *g = REAL(statx);
    if (!g) return NOREAL;
    int r = g(d, p, fl, mask, b);
    if (r == 0 && off) {
        int64_t m = shown(xns(&b->stx_mtime), 0, 1);
        if (b->stx_mask & STATX_ATIME) xput(&b->stx_atime, shown(xns(&b->stx_atime), m, 0));
        if (b->stx_mask & STATX_CTIME) xput(&b->stx_ctime, shown(xns(&b->stx_ctime), m, 0));
        if (b->stx_mask & STATX_BTIME) xput(&b->stx_btime, shown(xns(&b->stx_btime), m, 0));
        if (b->stx_mask & STATX_MTIME) xput(&b->stx_mtime, m);
    }
    return r;
}

/* File times as set. */

static int64_t stored(int64_t t) { return t >= clock0 ? t - off : t; }

static const struct timespec *unshift(const struct timespec in[2], struct timespec out[2]) {
    if (!in || !off) return in;
    for (int i = 0; i < 2; i++) {
        out[i] = in[i];
        if (in[i].tv_nsec != UTIME_NOW && in[i].tv_nsec != UTIME_OMIT)
            put(&out[i], stored(ns(in[i].tv_sec, in[i].tv_nsec)));
    }
    return out;
}

static const struct timeval *unshift_tv(const struct timeval in[2], struct timeval out[2]) {
    if (!in || !off) return in;
    for (int i = 0; i < 2; i++) {
        struct timespec ts;
        put(&ts, stored(ns(in[i].tv_sec, in[i].tv_usec * 1000LL)));
        out[i].tv_sec = ts.tv_sec;
        out[i].tv_usec = ts.tv_nsec / 1000;
    }
    return out;
}

int utimensat(int d, const char *p, const struct timespec t[2], int fl) {
    __typeof__(utimensat) *g = REAL(utimensat);
    struct timespec o[2];
    return g ? g(d, p, unshift(t, o), fl) : NOREAL;
}

int futimens(int fd, const struct timespec t[2]) {
    __typeof__(futimens) *g = REAL(futimens);
    struct timespec o[2];
    return g ? g(fd, unshift(t, o)) : NOREAL;
}

#define UTIMES(f, a) { __typeof__(f) *g = REAL(f); struct timeval o[2]; return g ? g(a, unshift_tv(t, o)) : NOREAL; }
int utimes(const char *p, const struct timeval t[2]) UTIMES(utimes, p)
int lutimes(const char *p, const struct timeval t[2]) UTIMES(lutimes, p)
int futimes(int fd, const struct timeval t[2]) UTIMES(futimes, fd)

int utime(const char *p, const struct utimbuf *t) {
    __typeof__(utime) *g = REAL(utime);
    if (!g) return NOREAL;
    if (!t || !off) return g(p, t);
    struct utimbuf o = {stored(t->actime * NS) / NS, stored(t->modtime * NS) / NS};
    return g(p, &o);
}

/* Kernel receive times. */

ssize_t recvmsg(int fd, struct msghdr *msg, int flags) {
    __typeof__(recvmsg) *g = REAL(recvmsg);
    if (!g) return NOREAL;
    ssize_t n = g(fd, msg, flags);
    if (n < 0 || !off || !msg->msg_control) return n;
    for (struct cmsghdr *c = CMSG_FIRSTHDR(msg); c; c = CMSG_NXTHDR(msg, c)) {
        if (c->cmsg_level != SOL_SOCKET) continue;
        if ((c->cmsg_type == SO_TIMESTAMP_OLD || c->cmsg_type == SO_TIMESTAMP_NEW)
            && c->cmsg_len >= CMSG_LEN(sizeof(struct timeval))) {
            struct timeval tv;
            struct timespec ts;
            memcpy(&tv, CMSG_DATA(c), sizeof tv);
            put(&ts, ns(tv.tv_sec, tv.tv_usec * 1000LL) + off);
            tv.tv_sec = ts.tv_sec;
            tv.tv_usec = ts.tv_nsec / 1000;
            memcpy(CMSG_DATA(c), &tv, sizeof tv);
        } else if ((c->cmsg_type == SO_TIMESTAMPNS_OLD || c->cmsg_type == SO_TIMESTAMPNS_NEW)
                   && c->cmsg_len >= CMSG_LEN(sizeof(struct timespec))) {
            struct timespec ts;
            memcpy(&ts, CMSG_DATA(c), sizeof ts);
            put(&ts, ns(ts.tv_sec, ts.tv_nsec) + off);
            memcpy(CMSG_DATA(c), &ts, sizeof ts);
        }
    }
    return n;
}

/* The boot time ps adds a process's start to (libproc2, from /proc/stat). */

struct stat_result *procps_stat_get(struct stat_info *info, enum stat_item item) {
    __typeof__(procps_stat_get) *g = REAL(procps_stat_get);
    struct stat_result *r = g ? g(info, item) : NULL;
    if (!r || item != STAT_SYS_TIME_OF_BOOT || !off) return r;
    static __thread struct stat_result copy;
    copy = *r;
    copy.result.ul_int += off / NS;
    return &copy;
}
