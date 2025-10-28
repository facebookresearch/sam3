# flake8: noqa

import inspect
from functools import wraps
from time import perf_counter

DO_TIMING = False
DISPLAY_LESS_PROGRESS = False
timer_dict = {}
counter = 0


def time(f):
    @wraps(f)
    def wrap(*args, **kw):
        if DO_TIMING:
            # Run function with timing
            ts = perf_counter()
            result = f(*args, **kw)
            te = perf_counter()
            tt = te - ts

            # Get function name
            arg_names = inspect.getfullargspec(f)[0]
            if arg_names[0] == "self" and DISPLAY_LESS_PROGRESS:
                return result
            elif arg_names[0] == "self":
                method_name = type(args[0]).__name__ + "." + f.__name__
            else:
                method_name = f.__name__

            # Record accumulative time in each function for analysis
            if method_name in timer_dict.keys():
                timer_dict[method_name] += tt
            else:
                timer_dict[method_name] = tt

            # If code is finished, display timing summary
            if method_name == "Evaluator.evaluate":
                print("")
                print("Timing analysis:")
                for key, value in timer_dict.items():
                    print("%-70s %2.4f sec" % (key, value))
            else:
                # Get function argument values for printing special arguments of interest
                arg_titles = ["tracker", "seq", "cls"]
                arg_vals = []
                for i, a in enumerate(arg_names):
                    if a in arg_titles:
                        arg_vals.append(args[i])
                arg_text = "(" + ", ".join(arg_vals) + ")"

                # Display methods and functions with different indentation.
                if arg_names[0] == "self":
                    print("%-74s %2.4f sec" % (" " * 4 + method_name + arg_text, tt))
                elif arg_names[0] == "test":
                    pass
                else:
                    global counter
                    counter += 1
                    print("%i %-70s %2.4f sec" % (counter, method_name + arg_text, tt))

            return result
        else:
            # If config["TIME_PROGRESS"] is false, or config["USE_PARALLEL"] is true, run functions normally without timing.
            return f(*args, **kw)

    return wrap
