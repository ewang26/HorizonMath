/*
 * Shell launcher for Codex tool processes.
 *
 * Modal's gVisor runtime does not permit bubblewrap to configure a nested
 * loopback interface. Codex therefore uses its filesystem sandbox with the
 * profile network flag enabled, while this launcher applies an irreversible
 * seccomp filter before the actual shell starts. Every shell descendant
 * inherits the filter and cannot create or operate sockets.
 */

#include <errno.h>
#include <libgen.h>
#include <seccomp.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <unistd.h>

static int deny_syscall(scmp_filter_ctx context, int syscall_number) {
    return seccomp_rule_add(
        context,
        SCMP_ACT_ERRNO(EPERM),
        syscall_number,
        0
    );
}

static int install_network_filter(void) {
    static const int denied_syscalls[] = {
        SCMP_SYS(socket),
        SCMP_SYS(socketpair),
        SCMP_SYS(connect),
        SCMP_SYS(bind),
        SCMP_SYS(listen),
        SCMP_SYS(accept),
        SCMP_SYS(accept4),
        SCMP_SYS(sendto),
        SCMP_SYS(sendmsg),
        SCMP_SYS(sendmmsg),
        SCMP_SYS(recvfrom),
        SCMP_SYS(recvmsg),
        SCMP_SYS(recvmmsg),
        SCMP_SYS(shutdown),
        SCMP_SYS(getsockname),
        SCMP_SYS(getpeername),
        SCMP_SYS(getsockopt),
        SCMP_SYS(setsockopt),
    };
    scmp_filter_ctx context = seccomp_init(SCMP_ACT_ALLOW);
    if (context == NULL) {
        return -1;
    }
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        seccomp_release(context);
        return -1;
    }
    for (size_t index = 0;
         index < sizeof(denied_syscalls) / sizeof(denied_syscalls[0]);
         index++) {
        if (deny_syscall(context, denied_syscalls[index]) != 0) {
            seccomp_release(context);
            return -1;
        }
    }
    if (seccomp_load(context) != 0) {
        seccomp_release(context);
        return -1;
    }
    seccomp_release(context);
    return 0;
}

int main(int argc, char **argv) {
    (void)argc;
    const char *name = basename(argv[0]);
    const char *target =
        strcmp(name, "sh") == 0 ? "/bin/sh.real" : "/bin/bash.real";

    if (install_network_filter() != 0) {
        perror("install_network_filter");
        return 125;
    }
    execv(target, argv);
    perror("execv");
    return 126;
}
