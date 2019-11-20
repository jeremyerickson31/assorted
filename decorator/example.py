import functools


def func_call_log(func):

    @functools.wraps(func)  # used to preserve information about the add_two_numbers function
    def wrapper_func_call_log(*args, **kwargs):

        print("calling the function: " + str(func.__name__))
        print("Positional Arguments: " + str(args))
        print("Keyword Arguments: " + str(kwargs))
        out = func(*args, **kwargs)

        return out

    return wrapper_func_call_log


@func_call_log  # causes add_two_numbers and its arguments to be passed to decorator function
def add_two_numbers_implicit(a, b):
    z = a + b
    return z


# this function will be wrapped at usage
def add_two_numbers_explicit(a, b):
    z = a + b
    return z


# main runtime chunk
if __name__ == "__main__":
    print("--------")
    print("Calling implicitly wrapped function")
    num1 = add_two_numbers_implicit(4, 5)  # num will be an integer = 9
    print(num1)

    print("--------")
    print("Calling explicitly wrapped function")
    num2 = func_call_log(add_two_numbers_explicit)  # num2 will be a function that gets called
    print(num2(4, 5))

