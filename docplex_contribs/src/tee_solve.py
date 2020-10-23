import tee
import os


def solve_with_tee(mdl, log_dir=None, log_filename=None, **solve_args):
    """ Solves a model with the tee module.
    sends output both to standard output and a log file.

    """
    log_filename = log_filename or f"temp_solve_tee_{mdl.name}"
    if not log_filename.endswith(".log"):
        log_filename += ".log"
    log_path = os.path.join(log_dir or '.', log_filename)
    # 1024 is important here, as default is 0
    tee_stream =  tee.StdoutTee(log_path, buff=1024)
    # override log_output with tee object
    solve_args['log_output'] = tee_stream
    with tee_stream:
        sol = mdl.solve(**solve_args)

        # need to flush at the end.
        tee_stream.flush()
    print(f"-- log file {log_path} overwritten")
    return sol


if __name__ == '__main__':
    from examples.delivery.modeling.nurses import build
    nm = build()
    solve_with_tee(nm)

